"""
Programmatic client for the IRSA SPHEREx Spectrophotometry Tool.

Reverse-engineered from a successful browser submission. The tool is a Firefly
CmdSrv-fronted service: we POST a `tableSearch` command to CmdSrv/async, which
spawns a UWS job that we then poll via the IVOA UWS protocol.

Auth note: The Firefly CmdSrv accepts our request, but the underlying slurm
submission requires a userKey that's been registered with the IRSA cluster.
Fresh requests.Session userKeys get rejected ("Failed to submit job to Slurm").
The fix is to load cookies from a real browser session (or eventually from an
authenticated IRSA login flow).

Quotas (from sp.html):
  - 1 target per submitted job (--point=RA,DEC,LABEL is single-source)
  - 2 concurrent jobs per user
  - Runtime ≈ 1.67 * N_overlapping_images + 50 s per source
"""

from __future__ import annotations

import http.cookiejar
import json
import time
import uuid
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import requests

CMDSRV_URL = "https://irsa.ipac.caltech.edu/applications/spherex/CmdSrv/async"
APP_URL = "https://irsa.ipac.caltech.edu/applications/spherex/"

UWS_NS = {"uws": "http://www.ivoa.net/xml/UWS/v1.0",
          "xlink": "http://www.w3.org/1999/xlink"}


@dataclass
class Target:
    ra: float          # deg, ICRS
    dec: float         # deg, ICRS
    label: str         # short identifier; goes into the result VOTable


@dataclass
class JobHandle:
    job_id: str
    job_url: str       # IVOA UWS endpoint
    target: Target
    raw: dict          # full submission response


def session_from_cookies(cookies_path: str | Path | None = None,
                         jsessionid: str | None = None,
                         usrkey: str | None = None,
                         ingresscookie: str | None = None) -> requests.Session:
    """Build a requests.Session with cookies that slurm will accept.

    Provide either:
      - `cookies_path` to a Netscape cookies.txt exported from a browser
        (e.g. via the "Get cookies.txt" extension), or
      - `jsessionid` + `usrkey` (+ optional `ingresscookie`) copied from DevTools.
        INGRESSCOOKIE controls load-balancer routing; without it the JSESSIONID
        may hit a backend that doesn't recognize it.
    """
    s = requests.Session()
    s.headers["User-Agent"] = (
        "spherex-evolved-stars/0.1 (adamginsburg@ufl.edu)"
    )
    if cookies_path:
        cj = http.cookiejar.MozillaCookieJar(str(cookies_path))
        cj.load(ignore_discard=True, ignore_expires=True)
        s.cookies.update(cj)
    for name, value in (("JSESSIONID", jsessionid),
                        ("usrkey", usrkey),
                        ("INGRESSCOOKIE", ingresscookie)):
        if value:
            s.cookies.set(name, value,
                          domain="irsa.ipac.caltech.edu",
                          path="/applications/spherex")
    if not (cookies_path or jsessionid or usrkey):
        # Anonymous session — slurm submission will likely fail.
        s.get(CMDSRV_URL, params={"cmd": "ping"}, timeout=30)
    return s


def _build_request(target: Target, bkg_region: int = 15) -> dict:
    """Inner Firefly ServerRequest for the SpectrophotometryProcessor.

    The META_INFO fields (form_submitTo, dataServiceOptions) and ffSessionId
    mirror what the Firefly UI sends; they appear to be required for the server
    to assign a UWS UUID and return jobInfo.jobUrl in the response.
    """
    job_id = uuid.uuid4().hex[:22]
    tbl_id = f"Spec-photo-tbl-tbl_id-{uuid.uuid4().hex[:4]}-{int(time.time()) % 100}"
    return {
        "RequestClass": "ServerRequest",
        "UserTargetWorldPt": f"{target.ra};{target.dec};EQ_J2000;{target.label}",
        "startIdx": 0,
        "pageSize": 2147483647,
        "bgEstimationRegion": str(bkg_region),
        "tbl_id": tbl_id,
        "startTime": "",
        "id": "SpectrophotometryProcessor",
        "endTime": "",
        "exposureTimeMode": "mjd",
        "CONE_AREA_KEY_RESERVED": "CONE",
        "shapeFit": "false",
        "ffSessionId": job_id,
        "META_INFO": {
            "jobId": job_id,
            "dataServiceOptions": json.dumps({
                "obsCoreDownloadProps": {"downloadType": "package"},
                "generateDownloadFileName": True,
                "DataProductFactoryOptions": {"datalinkDisableMoreDrop": True},
            }),
            "tbl_id": tbl_id,
            "form_submitTo": "/results-viewer",
            "title": "Spectrophotometry Targets",
        },
    }


def submit(target: Target,
           session: requests.Session,
           bkg_region: int = 15,
           retries: int = 1,
           retry_delay: float = 20.0) -> JobHandle:
    """Submit one target. Returns a JobHandle with the UWS jobUrl.

    Sometimes Firefly returns a "bare" response: phase=QUEUED but no
    jobInfo.jobUrl and a Firefly-internal jobId instead of a UUID. The job is
    actually queued server-side but we have no way to poll it. We retry the
    submission `retries` times with `retry_delay` seconds between attempts;
    each retry creates a new job (the server-queued bare-response one becomes
    an orphan), but at least we get a tracked one.
    """
    payload = {
        "cmd": "tableSearch",
        "request": json.dumps(_build_request(target, bkg_region=bkg_region)),
    }
    last_body = None
    for attempt in range(retries + 1):
        if attempt > 0:
            time.sleep(retry_delay)
        r = session.post(CMDSRV_URL, data=payload, timeout=60)
        r.raise_for_status()
        body = r.json()
        if body.get("phase") == "ERROR":
            err = body.get("errorSummary", {}).get("message", "<no message>")
            raise RuntimeError(f"Submission rejected ({target.label}): {err}")
        job_url = body.get("jobInfo", {}).get("jobUrl")
        job_id = body.get("jobId")
        if job_url:
            return JobHandle(job_id=job_id, job_url=job_url,
                             target=target, raw=body)
        last_body = body
    raise RuntimeError(
        f"No jobUrl after {retries+1} attempts ({target.label}); "
        f"last response jobId={last_body.get('jobId') if last_body else '?'}"
    )


def status(handle: JobHandle, session: requests.Session) -> dict:
    """Poll the UWS endpoint."""
    r = session.get(handle.job_url, timeout=15)
    r.raise_for_status()
    root = ET.fromstring(r.text)
    phase = root.findtext("uws:phase", default="UNKNOWN", namespaces=UWS_NS)
    results = []
    for res in root.findall("uws:results/uws:result", UWS_NS):
        results.append({
            "id": res.get("id"),
            "href": res.get("{http://www.w3.org/1999/xlink}href"),
        })
    error = root.findtext("uws:errorSummary/uws:message",
                          default=None, namespaces=UWS_NS)
    return {"phase": phase, "results": results, "error": error}


def is_terminal(phase: str) -> bool:
    return phase in ("COMPLETED", "ERROR", "ABORTED")


def download_results(status_dict: dict, out_dir: Path,
                     session: requests.Session,
                     name_prefix: str = "") -> list[Path]:
    """Download all result files. Returns paths."""
    out_dir.mkdir(exist_ok=True, parents=True)
    paths = []
    for r in status_dict["results"]:
        href = r["href"]
        rid = r["id"] or href.rsplit("/", 1)[-1] or "result.vot"
        name = f"{name_prefix}{rid}" if name_prefix else rid
        if not name.lower().endswith((".xml", ".vot", ".votable")):
            name = f"{name}.vot"
        dest = out_dir / name
        with session.get(href, stream=True, timeout=300) as resp:
            resp.raise_for_status()
            with open(dest, "wb") as f:
                for chunk in resp.iter_content(8192):
                    f.write(chunk)
        paths.append(dest)
    return paths

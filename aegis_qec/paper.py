from __future__ import annotations

import copy
import hashlib
import json
import os
import platform
import shutil
import subprocess
import zipfile
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any

from .project import _sha256_file, audit_research_project, load_research_project

_STIM_BIB = """@article{gidney2021stim,
  doi = {10.22331/q-2021-07-06-497},
  url = {https://doi.org/10.22331/q-2021-07-06-497},
  title = {Stim: a fast stabilizer circuit simulator},
  author = {Gidney, Craig},
  journal = {Quantum},
  volume = {5},
  pages = {497},
  year = {2021}
}
"""

_PYMATCHING_BIB = """@article{Higgott2025sparseblossom,
  doi = {10.22331/q-2025-01-20-1600},
  url = {https://doi.org/10.22331/q-2025-01-20-1600},
  title = {Sparse Blossom: correcting a million errors per core second with minimum-weight matching},
  author = {Higgott, Oscar and Gidney, Craig},
  journal = {Quantum},
  volume = {9},
  pages = {1600},
  year = {2025}
}
"""


def _version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def _latex_escape(value: Any) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(char, char) for char in str(value))


def _paragraphs(value: Any) -> str:
    parts = [
        part.strip()
        for part in str(value or "").split("\n\n")
        if part.strip()
    ]
    if not parts:
        return r"\emph{Not provided.}"
    return "\n\n".join(_latex_escape(part) for part in parts)


def _authors(project: dict[str, Any]) -> str:
    return r" \and ".join(
        _latex_escape(author["name"])
        for author in project["authors"]
    )


def _anonymize_project(project: dict[str, Any]) -> dict[str, Any]:
    value = copy.deepcopy(project)
    value["authors"] = [{"name": "Anonymous Authors"}]
    if "code_url" in value:
        value["code_url"] = "withheld for blind review"
    if "data_url" in value:
        value["data_url"] = "withheld for blind review"
    for experiment in value.get("experiments", []):
        if isinstance(experiment, dict) and experiment.get("manifest"):
            experiment["manifest"] = Path(
                str(experiment["manifest"])
            ).name
    for discovery in value.get("discoveries", []):
        if isinstance(discovery, dict) and discovery.get("manifest"):
            discovery["manifest"] = Path(
                str(discovery["manifest"])
            ).name
    for artifact in value.get("artifacts", []):
        if isinstance(artifact, dict) and artifact.get("path"):
            artifact["path"] = Path(str(artifact["path"])).name
        if isinstance(artifact, dict) and artifact.get("url"):
            artifact["url"] = "withheld for blind review"
    value["bibliography_files"] = [
        Path(str(path)).name
        for path in value.get("bibliography_files", [])
    ]
    return value


def _sanitize_audit(
    audit: dict[str, Any],
    *,
    anonymize: bool,
) -> dict[str, Any]:
    if not anonymize:
        return copy.deepcopy(audit)
    value = copy.deepcopy(audit)
    value["project_path"] = "research-project.json"
    for artifact in value.get("artifacts", {}).values():
        if artifact.get("source_path"):
            artifact["source_path"] = Path(artifact["source_path"]).name
    for experiment in value.get("experiments", []):
        if experiment.get("manifest_path"):
            experiment["manifest_path"] = Path(
                experiment["manifest_path"]
            ).name
    for discovery in value.get("discoveries", []):
        if discovery.get("manifest_path"):
            discovery["manifest_path"] = Path(
                discovery["manifest_path"]
            ).name
    for bibliography in value.get("bibliography", []):
        if bibliography.get("path"):
            bibliography["path"] = Path(bibliography["path"]).name
    lock = value.get("protocol_lock")
    if isinstance(lock, dict) and lock.get("path"):
        lock["path"] = Path(lock["path"]).name
    return value


def _hypotheses(project: dict[str, Any]) -> str:
    items = project.get("hypotheses", [])
    if not items:
        return r"\emph{No hypotheses declared.}"
    lines = [r"\begin{enumerate}"]
    for item in items:
        lines.append(
            r"\item \textbf{"
            + _latex_escape(item["id"])
            + r":} "
            + _latex_escape(item["text"])
        )
    lines.append(r"\end{enumerate}")
    return "\n".join(lines)


def _result_claims(project: dict[str, Any]) -> str:
    claims = [
        claim
        for claim in project.get("claims", [])
        if str(claim.get("type", "result")).lower()
        in {"result", "interpretation"}
    ]
    if not claims:
        return r"\emph{No result claims declared.}"
    lines = [r"\begin{itemize}"]
    for claim in claims:
        lines.append(
            r"\item \textbf{"
            + _latex_escape(claim["id"])
            + r":} "
            + _latex_escape(claim["text"])
        )
    lines.append(r"\end{itemize}")
    return "\n".join(lines)


def _claim_table(audit: dict[str, Any]) -> str:
    rows = []
    for claim in audit["claims"]:
        evidence = []
        for item in claim["evidence"]:
            label = str(item.get("artifact", ""))
            if item.get("json_pointer"):
                label += str(item["json_pointer"])
            if "actual" in item:
                label += " = " + repr(item["actual"])
            evidence.append(label)
        rows.append(
            "{} & {} & {} & {} \\\\".format(
                _latex_escape(claim["id"]),
                _latex_escape(claim["type"]),
                "PASS" if claim["passed"] else "FAIL",
                _latex_escape("; ".join(evidence) or "none"),
            )
        )
    return "\n".join(rows) or r"\multicolumn{4}{l}{No claims declared.} \\"


def _artifact_table(audit: dict[str, Any]) -> str:
    rows = []
    for artifact_id in sorted(audit["artifacts"]):
        artifact = audit["artifacts"][artifact_id]
        rows.append(
            "{} & {} & {} & {} \\\\".format(
                _latex_escape(artifact_id),
                _latex_escape(artifact.get("kind", "file")),
                _latex_escape(Path(artifact.get("source_path", "")).name),
                _latex_escape(str(artifact.get("sha256", "missing"))[:16]),
            )
        )
    return "\n".join(rows) or r"\multicolumn{4}{l}{No artifacts declared.} \\"


def _environment() -> dict[str, Any]:
    packages = [
        "aegis-qec",
        "numpy",
        "pymatching",
        "stim",
        "sinter",
        "h5py",
        "matplotlib",
        "scipy",
    ]
    return {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "packages": {name: _version(name) for name in packages},
        "github_sha": os.environ.get("GITHUB_SHA"),
    }


def _aegis_bib(*, anonymize: bool) -> str:
    version = _version("aegis-qec") or "source"
    year = datetime.now(timezone.utc).year
    if anonymize:
        return f"""@software{{aegis_qec,
  author = {{Anonymous}},
  title = {{Aegis QEC research platform}},
  version = {{{version}}},
  year = {{{year}}},
  note = {{Repository identity withheld for blind review}}
}}
"""
    return f"""@software{{aegis_qec,
  author = {{Bahri, Hamid}},
  title = {{Aegis QEC: reproducible quantum error-correction research platform}},
  version = {{{version}}},
  url = {{https://github.com/hamidbahri92/Aegis}},
  year = {{{year}}}
}}
"""


def _references(
    project_path: Path,
    project: dict[str, Any],
    *,
    anonymize: bool,
) -> str:
    blocks = [
        _aegis_bib(anonymize=anonymize),
        _STIM_BIB,
        _PYMATCHING_BIB,
    ]
    for value in project.get("bibliography_files", []):
        path = Path(value)
        if not path.is_absolute():
            path = project_path.parent / path
        blocks.append(path.read_text(encoding="utf-8"))
    return "\n\n".join(block.strip() for block in blocks if block.strip()) + "\n"


def _tex(project: dict[str, Any], audit: dict[str, Any]) -> str:
    paper = project.get("paper", {})
    keywords = project.get("keywords", paper.get("keywords", []))
    if isinstance(keywords, str):
        keywords = [part.strip() for part in keywords.split(",") if part.strip()]
    keyword_text = ", ".join(str(item) for item in keywords)

    return f"""\\documentclass[11pt]{{article}}
\\usepackage[margin=1in]{{geometry}}
\\usepackage{{booktabs}}
\\usepackage{{longtable}}
\\usepackage{{array}}
\\usepackage{{hyperref}}
\\usepackage{{microtype}}
\\usepackage{{enumitem}}
\\newcolumntype{{L}}[1]{{>{{\\raggedright\\arraybackslash}}p{{#1}}}}
\\hypersetup{{colorlinks=true,linkcolor=blue,citecolor=blue,urlcolor=blue}}

\\title{{{_latex_escape(project["title"])}}}
\\author{{{_authors(project)}}}
\\date{{}}

\\begin{{document}}
\\maketitle

\\begin{{abstract}}
{_paragraphs(paper.get("abstract"))}
\\end{{abstract}}

\\noindent\\textbf{{Keywords:}} {_latex_escape(keyword_text)}

\\section{{Research question}}
{_latex_escape(project["research_question"])}

\\section{{Hypotheses}}
{_hypotheses(project)}

\\section{{Statement of need}}
{_paragraphs(paper.get("statement_of_need"))}

\\section{{State of the field}}
{_paragraphs(paper.get("state_of_field"))}

The computational workflow can use Stim \\cite{{gidney2021stim}} and
PyMatching sparse blossom \\cite{{Higgott2025sparseblossom}}.

\\section{{Software and research design}}
{_paragraphs(paper.get("software_design"))}

\\section{{Methods}}
{_paragraphs(paper.get("methods"))}

The project was represented by a versioned Aegis research-project manifest.
Result and interpretation claims were audited against declared artifacts and,
where configured, JSON-pointer predicates.

\\section{{Results}}
{_paragraphs(paper.get("results_context"))}

{_result_claims(project)}

\\section{{Limitations}}
{_paragraphs(paper.get("limitations"))}

\\section{{Research impact}}
{_paragraphs(paper.get("impact"))}

\\section{{Data availability}}
{_paragraphs(paper.get("data_availability"))}

\\section{{Code availability}}
{_paragraphs(paper.get("code_availability"))}

\\section{{AI usage disclosure}}
{_paragraphs(paper.get("ai_usage_disclosure"))}

\\section{{Competing interests}}
{_paragraphs(paper.get("competing_interests", "None declared."))}

\\section{{Funding}}
{_paragraphs(paper.get("funding", "No funding statement provided."))}

\\appendix
\\section{{Claim-to-evidence audit}}
\\begin{{longtable}}{{L{{0.15\\linewidth}} L{{0.14\\linewidth}} L{{0.10\\linewidth}} L{{0.52\\linewidth}}}}
\\toprule
Claim & Type & Audit & Evidence \\\\
\\midrule
\\endhead
{_claim_table(audit)}
\\bottomrule
\\end{{longtable}}

\\section{{Artifact inventory}}
\\begin{{longtable}}{{L{{0.18\\linewidth}} L{{0.13\\linewidth}} L{{0.43\\linewidth}} L{{0.18\\linewidth}}}}
\\toprule
Artifact & Kind & File & SHA-256 prefix \\\\
\\midrule
\\endhead
{_artifact_table(audit)}
\\bottomrule
\\end{{longtable}}

\\section{{Reproducibility note}}
A passing Aegis audit establishes traceability and integrity of the declared
evidence. It does not by itself establish external validity, absence of bias,
or generalization beyond the sampled workloads.

\\bibliographystyle{{plain}}
\\bibliography{{references}}
\\end{{document}}
"""


def _paper_md(project: dict[str, Any]) -> str:
    paper = project.get("paper", {})
    lines = [
        "# " + project["title"],
        "",
        "Authors: " + ", ".join(a["name"] for a in project["authors"]),
        "",
        "## Summary",
        str(paper.get("abstract", "")).strip() or "Not provided.",
        "",
        "## Statement of need",
        str(paper.get("statement_of_need", "")).strip() or "Not provided.",
        "",
        "## State of the field",
        str(paper.get("state_of_field", "")).strip() or "Not provided.",
        "",
        "## Software design",
        str(paper.get("software_design", "")).strip() or "Not provided.",
        "",
        "## Methods",
        str(paper.get("methods", "")).strip() or "Not provided.",
        "",
        "## Results",
    ]
    for claim in project.get("claims", []):
        if str(claim.get("type", "result")).lower() in {"result", "interpretation"}:
            lines.append("- " + claim["id"] + ": " + claim["text"])
    lines.extend(
        [
            "",
            "## Limitations",
            str(paper.get("limitations", "")).strip() or "Not provided.",
            "",
            "## Research impact",
            str(paper.get("impact", "")).strip() or "Not provided.",
            "",
            "## Data availability",
            str(paper.get("data_availability", "")).strip() or "Not provided.",
            "",
            "## Code availability",
            str(paper.get("code_availability", "")).strip() or "Not provided.",
            "",
            "## AI usage disclosure",
            str(paper.get("ai_usage_disclosure", "")).strip() or "Not provided.",
            "",
        ]
    )
    return "\n".join(lines)


def _reviewer_readme(project: dict[str, Any], audit: dict[str, Any]) -> str:
    return f"""# Reviewer roadmap

Research package: {project["title"]}

Packaging-time claim/evidence audit: {"PASS" if audit["valid"] else "FAIL"}

Fast review path:

1. Read manuscript/paper.pdf when present, otherwise manuscript/paper.tex.
2. Read CLAIM_EVIDENCE.md for claim-to-artifact traceability.
3. Read audit.json for machine-readable evidence checks.
4. Read REPRODUCE.md for verification commands.
5. Inspect experiments/ and discoveries/ for the exact reviewed definitions.
6. Inspect artifacts/ for declared evidence.
7. Verify MANIFEST.json or checksums.sha256 before using artifacts.

Scientific boundary:

A passing audit means declared files exist, hashes were recorded, JSON pointers
resolve, and declared predicates hold. It does not certify an unbiased design,
a realistic device model, or generalization beyond the stated workloads.

Questions a strict reviewer should still ask:

- Does the research question match the experiment design?
- Were confirmatory hypotheses frozen before the relevant experiments?
- Are baselines and competing methods appropriate?
- Are all quantitative claims linked to artifacts?
- Are uncertainty intervals and paired tests appropriate?
- Are negative or null results reported?
- Are code, data, seeds, versions, and commands sufficient?
- Are limitations and external-validity boundaries explicit?
- Is AI assistance disclosed and human validation described?

Project SHA-256: {audit["project_sha256"]}
"""


def _reproduce(
    project_path: Path,
    project: dict[str, Any],
    audit: dict[str, Any],
) -> str:
    command = "aegis project audit " + project_path.name
    if audit.get("protocol_lock") is not None:
        command += " --require-protocol-lock"

    lines = [
        "# Reproduction and verification",
        "",
        "Verify the project:",
        "",
        "    " + command,
        "",
    ]

    commands = project.get("reproduction_commands", [])
    lines.extend(["Exact project reproduction commands:", ""])
    if commands:
        for item in commands:
            lines.append("    " + str(item))
    else:
        lines.append("    No project-level reproduction commands declared.")

    lines.extend(
        [
            "",
            "Verify individual experiment bundles:",
            "",
            "    aegis verify-bundle path/to/experiment.aegis.zip",
            "",
            "Verify reusable datasets:",
            "",
            "    aegis dataset inspect path/to/dataset.h5 --json",
            "",
            "Re-run a paired decoder comparison:",
            "",
            (
                "    aegis dataset evaluate path/to/dataset.h5 "
                "--decoder DECODER_A DECODER_B --split test --json"
            ),
            "",
            "Compute resources declared by the project:",
            "",
        ]
    )
    resources = project.get("compute_resources", [])
    if resources:
        for resource in resources:
            lines.append("    " + json.dumps(resource, sort_keys=True))
    else:
        lines.append("    None declared.")

    lines.extend(
        [
            "",
            "See environment.json for package and platform versions observed when",
            "this submission package was assembled. MANIFEST.json and",
            "checksums.sha256 cover the packaged payload.",
            "",
        ]
    )
    return "\n".join(lines)


def _claim_map(audit: dict[str, Any]) -> str:
    lines = [
        "# Claim-to-evidence map",
        "",
        "Overall audit: " + ("PASS" if audit["valid"] else "FAIL"),
        "",
    ]
    for claim in audit["claims"]:
        lines.extend(
            [
                "## " + claim["id"],
                "",
                "Type: " + claim["type"],
                "",
                claim["text"],
                "",
                "Audit: " + ("PASS" if claim["passed"] else "FAIL"),
                "",
            ]
        )
        for item in claim["evidence"]:
            label = "- artifact " + str(item.get("artifact", ""))
            if item.get("json_pointer"):
                label += "; pointer " + str(item["json_pointer"])
            if "actual" in item:
                label += "; observed " + repr(item["actual"])
            label += "; " + ("PASS" if item.get("passed") else "FAIL")
            lines.append(label)
        lines.append("")
    return "\n".join(lines)


def _checklist(project: dict[str, Any], audit: dict[str, Any]) -> str:
    paper = project.get("paper", {})
    checks = [
        ("Research question is explicit", bool(project.get("research_question"))),
        ("At least one author is declared", bool(project.get("authors"))),
        ("Project audit passes", bool(audit["valid"])),
        ("Statement of need is present", bool(paper.get("statement_of_need"))),
        ("State-of-field comparison is present", bool(paper.get("state_of_field"))),
        ("Design rationale is present", bool(paper.get("software_design"))),
        ("Limitations are present", bool(paper.get("limitations"))),
        ("Data availability is present", bool(paper.get("data_availability"))),
        ("Code availability is present", bool(paper.get("code_availability"))),
        ("AI usage disclosure is present", bool(paper.get("ai_usage_disclosure"))),
    ]
    lines = [
        "# Submission readiness checklist",
        "",
        "Preparation aid only; this is not a guarantee of venue compliance.",
        "",
    ]
    for label, passed in checks:
        lines.append("- [" + ("x" if passed else " ") + "] " + label)
    lines.extend(
        [
            "",
            "Venue-specific checks still required:",
            "",
            "- current page limit and official style",
            "- anonymity requirements",
            "- code/data upload size and artifact rules",
            "- ethics, impact, conflict, and funding requirements",
            "- a human author reading and approving every generated sentence",
        ]
    )
    return "\n".join(lines)


def _project_relative_path(project_path: Path, value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = project_path.parent / path
    return path.resolve()


def _copy_experiment_manifests(
    project_path: Path,
    project: dict[str, Any],
    root: Path,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    root.mkdir(parents=True, exist_ok=True)
    for experiment in project.get("experiments", []):
        experiment_id = str(experiment["id"])
        source = _project_relative_path(
            project_path,
            str(experiment["manifest"]),
        )
        if not source.is_file():
            continue
        target = root / (experiment_id + source.suffix)
        shutil.copy2(source, target)
        records.append(
            {
                "id": experiment_id,
                "packaged": str(target.relative_to(root.parent)),
                "sha256": _sha256_file(target),
                "bytes": target.stat().st_size,
            }
        )
    return records


def _copy_discovery_manifests(
    project_path: Path,
    project: dict[str, Any],
    root: Path,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    root.mkdir(parents=True, exist_ok=True)
    for discovery in project.get("discoveries", []):
        discovery_id = str(discovery["id"])
        source = _project_relative_path(
            project_path,
            str(discovery["manifest"]),
        )
        if not source.is_file():
            continue
        target = root / (discovery_id + source.suffix)
        shutil.copy2(source, target)
        records.append(
            {
                "id": discovery_id,
                "packaged": str(target.relative_to(root.parent)),
                "sha256": _sha256_file(target),
                "bytes": target.stat().st_size,
            }
        )
    return records


def _croissant_metadata(
    project: dict[str, Any],
    artifact_spec: dict[str, Any],
    artifact_record: dict[str, Any],
) -> dict[str, Any]:
    license_value = str(artifact_spec.get("license", "")).strip()
    url = str(artifact_spec.get("url", "")).strip()
    description = str(artifact_spec.get("description", "")).strip()
    if not license_value or not url or not description:
        raise ValueError(
            "Croissant export requires dataset description, license, and url "
            f"for artifact {artifact_spec.get('id')!r}"
        )
    return {
        "@context": {
            "@language": "en",
            "@vocab": "https://schema.org/",
            "sc": "https://schema.org/",
            "cr": "http://mlcommons.org/croissant/",
            "dct": "http://purl.org/dc/terms/",
            "conformsTo": "dct:conformsTo",
        },
        "@type": "sc:Dataset",
        "name": str(
            artifact_spec.get("name")
            or artifact_spec.get("id")
        ),
        "description": description,
        "license": license_value,
        "url": url,
        "conformsTo": "http://mlcommons.org/croissant/1.0",
        "distribution": [
            {
                "@type": "cr:FileObject",
                "@id": Path(artifact_record["packaged"]).name,
                "name": Path(artifact_record["packaged"]).name,
                "contentSize": f"{artifact_record['bytes']} B",
                "contentUrl": url,
                "encodingFormat": str(
                    artifact_spec.get(
                        "encoding_format",
                        "application/octet-stream",
                    )
                ),
                "sha256": artifact_record["sha256"],
            }
        ],
    }


def _write_croissant_metadata(
    project: dict[str, Any],
    artifact_inventory: list[dict[str, Any]],
    root: Path,
) -> list[dict[str, Any]]:
    by_id = {
        str(item["id"]): item
        for item in project.get("artifacts", [])
    }
    result: list[dict[str, Any]] = []
    for record in artifact_inventory:
        spec = by_id.get(record["id"], {})
        if str(spec.get("kind", "")).lower() != "dataset":
            continue
        requested = bool(spec.get("croissant", False))
        metadata_ready = all(
            str(spec.get(key, "")).strip()
            for key in ("description", "license", "url")
        )
        if not requested and not metadata_ready:
            continue
        metadata = _croissant_metadata(project, spec, record)
        target_dir = root / record["id"]
        target_dir.mkdir(parents=True, exist_ok=True)
        target = target_dir / "croissant.json"
        target.write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        result.append(
            {
                "artifact": record["id"],
                "packaged": str(target.relative_to(root.parent)),
                "sha256": _sha256_file(target),
                "bytes": target.stat().st_size,
            }
        )
    return result


def _submission_readiness(
    project: dict[str, Any],
    audit: dict[str, Any],
    *,
    croissant_records: list[dict[str, Any]],
) -> dict[str, Any]:
    paper = project.get("paper", {})
    venue = str(paper.get("venue", "generic")).strip().lower()
    protocol_mode = str(
        project.get("protocol", {}).get("mode", "exploratory")
    ).lower()
    artifact_specs = {
        str(item["id"]): item
        for item in project.get("artifacts", [])
    }
    dataset_ids = {
        artifact_id
        for artifact_id, item in artifact_specs.items()
        if str(item.get("kind", "")).lower() == "dataset"
    }
    croissant_ids = {
        str(item["artifact"])
        for item in croissant_records
    }

    checks: list[dict[str, Any]] = []

    def add(identifier: str, passed: bool, message: str) -> None:
        checks.append(
            {
                "id": identifier,
                "passed": bool(passed),
                "message": message,
            }
        )

    add(
        "claim_evidence",
        bool(audit.get("valid")),
        "All declared scientific claims and artifacts pass the project audit.",
    )
    add(
        "confirmatory_protocol_lock",
        protocol_mode != "confirmatory"
        or bool(audit.get("protocol_lock", {}).get("valid")),
        "Confirmatory projects require a valid frozen protocol.",
    )
    add(
        "reproduction_commands",
        bool(project.get("reproduction_commands")),
        "Exact reproduction commands are declared.",
    )
    add(
        "code_availability",
        bool(str(project.get("code_url", "")).strip())
        or bool(str(paper.get("code_availability", "")).strip()),
        "Code access or a code-availability statement is declared.",
    )
    add(
        "data_availability",
        bool(str(project.get("data_url", "")).strip())
        or bool(str(paper.get("data_availability", "")).strip()),
        "Data access or a data-availability statement is declared.",
    )
    add(
        "limitations",
        bool(str(paper.get("limitations", "")).strip()),
        "Limitations are explicitly documented.",
    )
    add(
        "ai_usage",
        bool(str(paper.get("ai_usage_disclosure", "")).strip()),
        "Generative-AI use is explicitly disclosed.",
    )
    add(
        "bibliography",
        bool(project.get("bibliography_files")),
        "At least one project bibliography file is declared.",
    )

    if venue == "joss":
        for key, label in [
            ("statement_of_need", "JOSS statement of need is present."),
            ("state_of_field", "JOSS state of the field is present."),
            ("software_design", "JOSS software design section is present."),
            ("impact", "JOSS research impact statement is present."),
        ]:
            add(
                "joss_" + key,
                bool(str(paper.get(key, "")).strip()),
                label,
            )

    if venue in {"neurips", "neurips-ed"}:
        add(
            "compute_resources",
            bool(project.get("compute_resources")),
            "Experimental compute resources are documented.",
        )
        methods_text = str(paper.get("methods", "")).lower()
        statistical_terms = (
            "confidence interval",
            "wilson",
            "bootstrap",
            "standard error",
            "p-value",
            "p value",
            "mcnemar",
            "significance",
        )
        add(
            "statistical_uncertainty",
            any(term in methods_text for term in statistical_terms),
            "Methods document uncertainty intervals or significance testing.",
        )

    if venue == "neurips-ed" and dataset_ids:
        add(
            "croissant",
            dataset_ids <= croissant_ids,
            "Every dataset contribution has Croissant 1.0 metadata.",
        )

    failed = [item for item in checks if not item["passed"]]
    return {
        "schema_version": 1,
        "venue": venue,
        "ready": not failed,
        "checks": checks,
        "failures": failed,
    }


def _redacted_protocol_lock(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    protocol = copy.deepcopy(value.get("project_protocol", {}))
    for experiment in protocol.get("experiments", []):
        if isinstance(experiment, dict) and experiment.get("manifest"):
            experiment["manifest"] = Path(
                str(experiment["manifest"])
            ).name
    for discovery in protocol.get("discoveries", []):
        if isinstance(discovery, dict) and discovery.get("manifest"):
            discovery["manifest"] = Path(
                str(discovery["manifest"])
            ).name

    return {
        "schema_version": value.get("schema_version", 1),
        "lock_type": value.get("lock_type"),
        "created_utc": value.get("created_utc"),
        "project_title_at_freeze": value.get("project_title_at_freeze"),
        "project_protocol": protocol,
        "project_protocol_sha256": value.get("project_protocol_sha256"),
        "protocol_sha256": value.get("protocol_sha256"),
        "redacted": True,
        "files": [
            {
                "role": item.get("role"),
                "path": Path(str(item.get("path", ""))).name,
                "sha256": item.get("sha256"),
            }
            for item in value.get("files", [])
        ],
    }

def _copy_artifacts(
    audit: dict[str, Any],
    root: Path,
    *,
    anonymize: bool,
) -> list[dict[str, Any]]:
    records = []
    for artifact_id in sorted(audit["artifacts"]):
        artifact = audit["artifacts"][artifact_id]
        if not artifact.get("exists"):
            continue
        source = Path(artifact["source_path"])
        target_dir = root / artifact_id
        target_dir.mkdir(parents=True, exist_ok=True)
        target = target_dir / source.name
        shutil.copy2(source, target)
        record = {
            "id": artifact_id,
            "packaged": str(target.relative_to(root.parent)),
            "sha256": _sha256_file(target),
            "bytes": target.stat().st_size,
        }
        if not anonymize:
            record["source"] = str(source)
        records.append(record)
    return records


def _compile(manuscript_dir: Path, mode: str) -> dict[str, Any]:
    if mode not in {"auto", "never", "required"}:
        raise ValueError("compile mode must be auto, never, or required")
    if mode == "never":
        return {"requested": False, "compiled": False, "engine": None}

    choices = []
    if shutil.which("tectonic"):
        choices.append(
            ("tectonic", ["tectonic", "--keep-logs", "paper.tex"])
        )
    if shutil.which("latexmk"):
        choices.append(
            (
                "latexmk",
                [
                    "latexmk",
                    "-pdf",
                    "-interaction=nonstopmode",
                    "-halt-on-error",
                    "paper.tex",
                ],
            )
        )
    if not choices:
        if mode == "required":
            raise RuntimeError(
                "No supported LaTeX compiler found; install tectonic or latexmk."
            )
        return {
            "requested": True,
            "compiled": False,
            "engine": None,
            "reason": "compiler unavailable",
        }

    engine, command = choices[0]
    result = subprocess.run(
        command,
        cwd=manuscript_dir,
        capture_output=True,
        text=True,
        check=False,
        timeout=300,
    )
    log_path = manuscript_dir / "aegis-latex-build.log"
    log_path.write_text(
        "$ "
        + " ".join(command)
        + "\n\n"
        + result.stdout
        + "\n--- stderr ---\n"
        + result.stderr,
        encoding="utf-8",
    )
    compiled = result.returncode == 0 and (manuscript_dir / "paper.pdf").is_file()
    if not compiled and mode == "required":
        raise RuntimeError(
            "LaTeX compilation failed; see " + str(log_path)
        )
    return {
        "requested": True,
        "compiled": compiled,
        "engine": engine,
        "returncode": result.returncode,
        "log": str(log_path),
    }


def _payload(root: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(root).as_posix()
        if rel in {"MANIFEST.json", "checksums.sha256"}:
            continue
        rows.append(
            {
                "path": rel,
                "sha256": _sha256_file(path),
                "bytes": path.stat().st_size,
            }
        )
    return rows


def _zip_directory(source: Path, target: Path) -> None:
    with zipfile.ZipFile(target, "w") as archive:
        for path in sorted(source.rglob("*")):
            if not path.is_file():
                continue
            info = zipfile.ZipInfo(path.relative_to(source).as_posix())
            info.date_time = (1980, 1, 1, 0, 0, 0)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            archive.writestr(info, path.read_bytes())


def build_submission_package(
    project_path: str,
    *,
    output_dir: str,
    compile_mode: str = "auto",
    require_protocol_lock: bool = False,
    allow_invalid: bool = False,
    anonymize: bool = False,
    overwrite: bool = False,
) -> dict[str, Any]:
    source = Path(project_path).resolve()
    project = load_research_project(str(source))
    audit = audit_research_project(
        str(source),
        require_protocol_lock=require_protocol_lock,
    )
    if not audit["valid"] and not allow_invalid:
        raise ValueError(
            "research project audit failed: " + "; ".join(audit["errors"])
        )

    destination = Path(output_dir).resolve()
    zip_path = Path(str(destination) + ".zip")
    if destination.exists() or zip_path.exists():
        if not overwrite:
            raise FileExistsError(
                "submission output already exists; choose a new path or use overwrite"
            )
        if destination.exists():
            shutil.rmtree(destination)
        if zip_path.exists():
            zip_path.unlink()
    destination.mkdir(parents=True)
    manuscript = destination / "manuscript"
    artifacts = destination / "artifacts"
    experiments = destination / "experiments"
    discoveries = destination / "discoveries"
    dataset_metadata = destination / "dataset-metadata"
    manuscript.mkdir()
    artifacts.mkdir()
    experiments.mkdir()
    discoveries.mkdir()
    dataset_metadata.mkdir()

    packaged_project = (
        _anonymize_project(project)
        if anonymize
        else copy.deepcopy(project)
    )
    packaged_audit = _sanitize_audit(audit, anonymize=anonymize)
    (destination / "research-project.json").write_text(
        json.dumps(packaged_project, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    lock = source.with_suffix(".protocol.lock.json")
    if lock.is_file():
        packaged_lock = destination / "research-project.protocol.lock.json"
        if anonymize:
            packaged_lock.write_text(
                json.dumps(
                    _redacted_protocol_lock(lock),
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
        else:
            shutil.copy2(lock, packaged_lock)

    (destination / "audit.json").write_text(
        json.dumps(packaged_audit, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (destination / "environment.json").write_text(
        json.dumps(_environment(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (destination / "CLAIM_EVIDENCE.md").write_text(
        _claim_map(packaged_audit) + "\n",
        encoding="utf-8",
    )
    (destination / "REVIEWER_README.md").write_text(
        _reviewer_readme(packaged_project, packaged_audit) + "\n",
        encoding="utf-8",
    )
    (destination / "REPRODUCE.md").write_text(
        _reproduce(
            Path("research-project.json"),
            packaged_project,
            packaged_audit,
        )
        + "\n",
        encoding="utf-8",
    )

    experiment_inventory = _copy_experiment_manifests(
        source,
        project,
        experiments,
    )
    (destination / "experiment-inventory.json").write_text(
        json.dumps(experiment_inventory, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    discovery_inventory = _copy_discovery_manifests(
        source,
        project,
        discoveries,
    )
    (destination / "discovery-inventory.json").write_text(
        json.dumps(discovery_inventory, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    inventory = _copy_artifacts(
        audit,
        artifacts,
        anonymize=anonymize,
    )
    (destination / "artifact-inventory.json").write_text(
        json.dumps(inventory, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    croissant = _write_croissant_metadata(
        project,
        inventory,
        dataset_metadata,
    )
    (destination / "croissant-inventory.json").write_text(
        json.dumps(croissant, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    readiness = _submission_readiness(
        project,
        audit,
        croissant_records=croissant,
    )
    (destination / "submission-readiness.json").write_text(
        json.dumps(readiness, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    checklist = _checklist(packaged_project, packaged_audit)
    if readiness["failures"]:
        checklist += "\n\nMachine-audited readiness failures:\n\n"
        for item in readiness["failures"]:
            checklist += "- [ ] " + item["message"] + "\n"
    else:
        checklist += "\n\nMachine-audited readiness: PASS\n"
    (destination / "SUBMISSION_CHECKLIST.md").write_text(
        checklist + "\n",
        encoding="utf-8",
    )

    (manuscript / "paper.tex").write_text(
        _tex(packaged_project, packaged_audit),
        encoding="utf-8",
    )
    (manuscript / "paper.md").write_text(
        _paper_md(packaged_project) + "\n",
        encoding="utf-8",
    )
    (manuscript / "references.bib").write_text(
        _references(
            source,
            project,
            anonymize=anonymize,
        ),
        encoding="utf-8",
    )

    compile_result = _compile(manuscript, compile_mode)
    (destination / "latex-build.json").write_text(
        json.dumps(compile_result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    payload = _payload(destination)
    manifest = {
        "schema_version": 1,
        "package_type": "aegis_qec_submission_package",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "project_title": packaged_project["title"],
        "project_sha256": audit["project_sha256"],
        "audit_valid": audit["valid"],
        "submission_ready": readiness["ready"],
        "submission_readiness": readiness,
        "anonymized": bool(anonymize),
        "experiment_count": len(experiment_inventory),
        "discovery_count": len(discovery_inventory),
        "croissant_count": len(croissant),
        "payload": payload,
    }
    manifest_path = destination / "MANIFEST.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    checks = payload + [
        {
            "path": "MANIFEST.json",
            "sha256": _sha256_file(manifest_path),
            "bytes": manifest_path.stat().st_size,
        }
    ]
    (destination / "checksums.sha256").write_text(
        "".join(
            item["sha256"] + "  " + item["path"] + "\n"
            for item in checks
        ),
        encoding="utf-8",
    )

    _zip_directory(destination, zip_path)

    return {
        "schema_version": 1,
        "package_type": "aegis_qec_submission_package",
        "path": str(destination),
        "zip_path": str(zip_path),
        "zip_sha256": _sha256_file(zip_path),
        "audit_valid": audit["valid"],
        "submission_ready": readiness["ready"],
        "submission_readiness": readiness,
        "compile": compile_result,
        "artifact_count": len(inventory),
        "experiment_count": len(experiment_inventory),
        "discovery_count": len(discovery_inventory),
        "croissant_count": len(croissant),
        "claim_count": len(audit["claims"]),
    }




def verify_submission_package(path: str) -> dict[str, Any]:
    """Verify a submission archive or directory without trusting manifest paths.

    Content hashes and the checksums file establish internal consistency only;
    they are not a digital signature of the submitter.
    """
    from .archive_integrity import verify_manifested_payload

    source = Path(path).resolve()
    failures = verify_manifested_payload(
        source,
        manifest_name="MANIFEST.json",
        manifest_type="aegis_qec_submission_package",
        inventory_name="payload",
        required_names=("research-project.json", "audit.json"),
        has_checksums=True,
    )
    return {
        "schema_version": 1,
        "verification_type": "aegis_qec_submission_package",
        "path": str(source),
        "valid": not failures,
        "failures": failures,
        "zip_sha256": _sha256_file(source) if source.is_file() else None,
    }

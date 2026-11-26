#!/bin/bash
# Run from the deploy/ folder: registers the Orthanc auto-routing rule, then listens for studies.

curl -X POST http://localhost:8042/tools/execute-script --data-binary @route_dicoms.lua -v

# DICOM route directory: one folder per study is created here; pass it to inference_dcm.py.
ROUTED_DIR="${ROUTED_DIR:-$(cd .. && pwd)/outputs/routed}"
mkdir -p "$ROUTED_DIR"
# port 106 needs root; prefix with sudo if you are not root and sudo can find storescp
storescp 106 -v -aet HIPPOAI -od "$ROUTED_DIR" --sort-on-study-uid st

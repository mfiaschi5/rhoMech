"""ksmech-serve: the local web application for predicting, inspecting, correcting and curating
mechanisms (backend). This module imports nothing, so `ksmech.serve.app.cli` can pin the BLAS and
OpenMP threads before numpy loads."""

API_SCHEMA = 1

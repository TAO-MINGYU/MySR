# This dockerfile simulates a user installation that
# tries to manually edit SymbolicRegression.jl and
# use it from MySR.

ARG JLVERSION=1.11
ARG PYVERSION=3.12
ARG BASE_IMAGE=bullseye

FROM julia:${JLVERSION}-${BASE_IMAGE} AS jl
FROM python:${PYVERSION}-${BASE_IMAGE}

# Merge Julia image:
COPY --from=jl /usr/local/julia /usr/local/julia
ENV PATH="/usr/local/julia/bin:${PATH}"

WORKDIR /mysr

# Install MySR:
# We do a minimal copy so it doesn't need to rerun at every file change:
ADD ./pyproject.toml /mysr/pyproject.toml
ADD ./LICENSE /mysr/LICENSE
ADD ./README.md /mysr/README.md

RUN mkdir /mysr/mysr
ADD ./mysr/*.py /mysr/mysr/
ADD ./mysr/juliapkg.json /mysr/mysr/juliapkg.json

RUN mkdir /mysr/mysr/_cli
ADD ./mysr/_cli/*.py /mysr/mysr/_cli/

RUN mkdir /mysr/mysr/test

# Now, we create a custom version of SymbolicRegression.jl
# First, we get the version or rev from juliapkg.json:
RUN python3 -c 'import json; pkg = json.load(open("/mysr/mysr/juliapkg.json", "r"))["packages"]["SymbolicRegression"]; print(pkg.get("version", pkg.get("rev", "")))' > /mysr/sr_version

# Remove any = or ^ or ~ from the version:
RUN cat /mysr/sr_version | sed 's/[\^=~]//g' > /mysr/sr_version_processed

# Now, we check out the version of SymbolicRegression.jl that MySR is using:
# If sr_version starts with 'v', use it as-is; otherwise prepend 'v'
RUN if grep -q '^v' /mysr/sr_version_processed; then \
        git clone -b "$(cat /mysr/sr_version_processed)" --single-branch https://github.com/astroautomata/SymbolicRegression.jl /srjl; \
    else \
        git clone -b "v$(cat /mysr/sr_version_processed)" --single-branch https://github.com/astroautomata/SymbolicRegression.jl /srjl; \
    fi

# Edit SymbolicRegression.jl to create a new function.
# We want to put this function immediately after `module SymbolicRegression`:
RUN sed -i 's/module SymbolicRegression/module SymbolicRegression\n__test_function() = 2.3/' /srjl/src/SymbolicRegression.jl

# Edit MySR to use the custom version of SymbolicRegression.jl:
ADD ./mysr/test/generate_dev_juliapkg.py /generate_dev_juliapkg.py
RUN python3 /generate_dev_juliapkg.py /mysr/mysr/juliapkg.json /srjl

# Install and pre-compile
RUN pip3 install --no-cache-dir . && python3 -c 'import mysr'

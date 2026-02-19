# Multi-stage build for small image size
FROM continuumio/miniconda3:24.1.2-0 AS base

WORKDIR /app

# Install RDKit via conda (system rdkit is most reliable)
RUN conda install -y -c conda-forge rdkit=2023.09.6 && \
    conda clean -afy

# Install molscreen
COPY pyproject.toml README.md ./
COPY molscreen/ ./molscreen/
RUN pip install --no-cache-dir -e . && \
    pip cache purge

# Runtime stage
FROM base AS runtime
WORKDIR /data
COPY --from=base /opt/conda /opt/conda
COPY --from=base /app /app
ENV PATH="/opt/conda/bin:${PATH}"
WORKDIR /data

ENTRYPOINT ["molscreen"]
CMD ["--help"]

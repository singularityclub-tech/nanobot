FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

# # Install Node.js 20 for the WhatsApp bridge
# RUN apt-get update && \
#     apt-get install -y --no-install-recommends curl ca-certificates gnupg git bubblewrap openssh-client && \
#     mkdir -p /etc/apt/keyrings && \
#     curl -fsSL https://deb.nodesource.com/gpgkey/nodesource-repo.gpg.key | gpg --dearmor -o /etc/apt/keyrings/nodesource.gpg && \
#     echo "deb [signed-by=/etc/apt/keyrings/nodesource.gpg] https://deb.nodesource.com/node_20.x nodistro main" > /etc/apt/sources.list.d/nodesource.list && \
#     apt-get update && \
#     apt-get install -y --no-install-recommends nodejs && \
#     apt-get purge -y gnupg && \
#     apt-get autoremove -y && \
#     rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml uv.lock README.md LICENSE ./
COPY nanobot/ nanobot/
COPY bridge/ bridge/
COPY singularity-cli/ singularity-cli/
COPY singularity-client/ singularity-client/
RUN uv pip install --system --no-cache ./singularity-client ./singularity-cli .
RUN printf '%s\n' '#!/bin/sh' 'export PYTHONPATH="/app/singularity-cli/src:/app/singularity-client${PYTHONPATH:+:$PYTHONPATH}"' 'exec python3 -m singularity_cli.cli "$@"' > /usr/local/bin/sg && \
    chmod +x /usr/local/bin/sg

# # Build the WhatsApp bridge
# WORKDIR /app/bridge
# RUN git config --global --add url."https://github.com/".insteadOf ssh://git@github.com/ && \
#     git config --global --add url."https://github.com/".insteadOf git@github.com: && \
#     npm install && npm run build
# WORKDIR /app

# # Create non-root user and config directory
RUN useradd -m -u 1000 -s /bin/bash nanobot && \
    mkdir -p /home/nanobot/.nanobot && \
    chown -R nanobot:nanobot /home/nanobot /app

COPY entrypoint.sh /usr/local/bin/entrypoint.sh
RUN sed -i 's/\r$//' /usr/local/bin/entrypoint.sh && chmod +x /usr/local/bin/entrypoint.sh

USER nanobot
ENV HOME=/home/nanobot
# Gateway default port
EXPOSE 18790

ENTRYPOINT ["entrypoint.sh"]
CMD ["status"]

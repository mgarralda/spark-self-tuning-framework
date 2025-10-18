# -----------------------------------------------------------------------------
#  Project: Spark Self-Tuning Framework (STL-PARN-ILS-TS-BO)
#  File: yoro_perf_model_runner.py
#  Copyright (c) 2025 Mariano Garralda Barrio
#  Affiliation: Universidade da Coruña
#  SPDX-License-Identifier: CC-BY-NC-4.0 OR LicenseRef-Commercial
#
#  Associated publication:
#    "A hybrid metaheuristics–Bayesian optimization framework with safe transfer learning for continuous Spark tuning"
#    Mariano Garralda Barrio, Verónica Bolón Canedo, Carlos Eiras Franco
#    Universidade da Coruña, 2025.
#
#  Academic & research use: CC BY-NC 4.0
#    https://creativecommons.org/licenses/by-nc/4.0/
#  Commercial use: requires prior written consent.
#    Contact: mariano.garralda@udc.es
#
#  Distributed on an "AS IS" basis, without warranties or conditions of any kind.
# -----------------------------------------------------------------------------

import asyncio
from typing import Generator
import paramiko
from config import SSHConfig
import asyncssh


class SSHClientManager:
    def __init__(self, ssh_config: SSHConfig):
        self.ssh_config = ssh_config
        self.client = paramiko.SSHClient()

    def __enter__(self):
        self.client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        self.client.connect(
            hostname=self.ssh_config.hostname,
            port=self.ssh_config.port,
            username=self.ssh_config.username,
            password=self.ssh_config.password
        )
        return self.client

    def __exit__(self, exc_type, exc_value, traceback):
        self.client.close()

class AsyncSSHClientManager:
    def __init__(self, ssh_config: SSHConfig):
        self.ssh_config = ssh_config
        self.conn = None

    async def __aenter__(self):
        self.conn = await asyncssh.connect(
            host=self.ssh_config.hostname,
            port=self.ssh_config.port,
            username=self.ssh_config.username,
            password=self.ssh_config.password,
            known_hosts=None  # disables known_hosts checking (like Paramiko's AutoAddPolicy)
        )
        return self.conn

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        self.conn.close()
        await self.conn.wait_closed()

def connect_via_ssh(ssh_config: SSHConfig) -> Generator[str, None, None]:
    with SSHClientManager(ssh_config) as client:
        stdin, stdout, stderr = client.exec_command(ssh_config.command)
        for line in iter(stdout.readline, ""):
            yield line

async def connect_via_async_ssh(ssh_config: SSHConfig):
    async with AsyncSSHClientManager(ssh_config) as conn:
        process = await conn.create_process(ssh_config.command)
        async for line in process.stdout:
            yield line

if __name__ == "__main__":
    ssh_config = SSHConfig()

    async def main():
        async for line in connect_via_async_ssh(ssh_config):
            print(line, end="")

    asyncio.run(main())

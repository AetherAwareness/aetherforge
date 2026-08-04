# Security Policy

## Supported versions

| Version | Supported |
|---------|-----------|
| 0.2.x   | Yes |
| < 0.2   | Best effort |

## Reporting a vulnerability

Please **do not** open a public issue for security-sensitive reports.

1. Email the maintainers (replace with your contact), **or**  
2. Use GitHub **Private vulnerability reporting** if enabled on the repository.

Include:

- AetherForge version (`aetherforge version`)  
- Steps to reproduce  
- Impact (data exposure, remote RCE, credential leak, etc.)  
- Whether a fix is already known  

We will acknowledge receipt and work on a fix or mitigation timeline.

## Non-security notes

- Remote train is intentionally gated (`--exec`, desktop YES).  
- Credentials belong in `~/.aetherforge/` (not the git tree).  
- Model weights and private corpora should not be committed.

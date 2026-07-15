# Full PETAL Evidence Trust Model

## Purpose

Full PETAL evidence supports scientific reproducibility. It proves that a
declared, source-locked training process followed the fail-closed lifecycle
wiring encoded by this repository and makes later artifact modification
detectable. It is not a hostile-process remote-attestation system.

## Trusted Computing Base

The evidence chain trusts these components for the duration of a validated
launch:

1. `launch_validator`: validates B0, independent review, launch identity, and
   the applicable profile or formal-training authorization before CUDA/DDP.
2. `train_engine`: owns the optimizer and online-state transaction lifecycle.
3. `runtime_evidence_session`: executes and signs the optimizer-step and
   transaction-commit boundary receipts before publishing an event.
4. `in_process_attestation_key_material`: the ephemeral Ed25519 private key
   held by the validated launch process.

The exact machine-readable representation is locked in
`launch_contract.evidence_trust_model`. Launch validation rejects any missing,
changed, weakened, or expanded representation.

## Guarantees

Within the trusted computing base, the evidence chain provides:

- `fail_closed_lifecycle_wiring`: an optimizer event cannot be published until
  the bound optimizer step and online-state transaction commit have completed.
- `provenance_binding`: signed receipts bind the launch session, runtime chain,
  boundary nonce, object identities, operation order, and prior receipt hash.
- `post_publication_tamper_evidence`: modification or replay of published
  receipts and event records fails verification.

## Explicit Non-Guarantees

The evidence chain does not claim protection from:

- `arbitrary_code_execution_inside_tcb`.
- `in_process_private_key_compromise` or deliberate misuse of that key by code
  already executing inside the trusted process.

Those claims require an isolated signer such as an HSM, enclave, or separately
privileged service and are outside this scientific-reproducibility route. If
key compromise is suspected, the required action is
`BLOCK_ROTATE_AND_RERUN`; affected evidence is not admissible.

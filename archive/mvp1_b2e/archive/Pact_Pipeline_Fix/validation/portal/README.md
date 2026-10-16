# Portal launch pipeline test

The seven production modules were tested unchanged.

## Original Teams input

Automatic format detection returned `plain_text`. Parsing produced two units with no author labels instead of 68 messages from seven participants. Native ingestion therefore fails this fixture.

## Converted-input diagnostic

The standalone runner converts only Teams message headers to the supported WhatsApp syntax, preserving authors, timestamps, message text and sequence. It then runs the actual patched extractor, merger, reconciliation and deterministic renderer. This is a diagnostic of downstream behavior, not a native Teams end-to-end pass.

The anniversary scoring function is disabled: its names, options and expectations do not apply to this transcript. The portal output is reviewed separately against the expected card supplied by the user.

No production code was changed during this test. Credentials are excluded from this package.

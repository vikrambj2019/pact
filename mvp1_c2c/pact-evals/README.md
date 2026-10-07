# Pact decision-engine eval set: hiking

Five synthetic group chats about the same November hiking trip, six people each (Sam, Alex, Priya, Jordan, Maya, Leo), rising from clean to chaotic. Each has a transcript and a gold label describing the correct decision state.

| # | File stem | Messages | What it adds |
| --- | --- | --- | --- |
| 1 | hiking-01-simple | 60 (10 each) | Clean and linear. Options, dates, budget, two withdrawals, explicit unanimous choice, next steps. |
| 2 | hiking-02-one-dispute | 66 | One disputed factual claim, one changed mind, side chatter, a quiet member (4 messages) who is never counted as agreeing. No decision. |
| 3 | hiking-03-shifting-constraints | 80 | Budget superseded mid-chat, dates contested by one member, an option dropped then revived, an ambiguous "Sounds good", a disputed calculation, a conditional drop. No decision. |
| 4 | hiking-04-interleaved-threads | 96 | Three decisions at once (destination, lodging, transport), nicknames, sarcasm, a misquote and a retraction, a claim about a person's stance that the person contradicts, a member who leaves and returns and reopens a decided question. |
| 5 | hiking-05-chaos | 123 | Everyone raising different points and claims, off-topic chat, media and deleted-message placeholders, half-serious options, conflicting claims with different source quality, a false "winning" claim, a poll with an abstention and a conditional vote, two side decisions started early, and two sensitive disclosures. No decision. |

## Files

* `out/<stem>.txt` is a WhatsApp Android-style export: `9/14/26, 7:40 PM - Sam: text`. The first line is the end-to-end encryption notice a real export starts with; a parser must skip it. Placeholders such as `<Media omitted>` and `This message was deleted` appear as real exports render them. Reply links are not in the text export (real exports don't carry them).
* `out/<stem>.json` is the same chat as structured data: message ids `m001...`, ISO timestamps (America/Los_Angeles), sender, text, and `reply_to` where a message replies to another. Use this to test a Telegram-like input that has reply metadata.
* `gold/<stem>.gold.json` is the expected decision state. Every entry cites message ids, so provenance can be scored.
* `src/<stem>.src` is the editable source. `python3 build.py` regenerates `out/` and checks ids, sender counts, reply links, and that every id in the gold files exists.

## Gold fields

`options`, `constraints`, `criteria`, `claims`, `agreements` (with `explicit_affirmers`), `open_issues`, `outcome`, `next_steps`, `traps` (what is deliberately tricky and the correct handling), `card_must_include`, `card_must_not_include`. Transcripts 2 to 5 also carry `preferences`; 4 and 5 add `aliases` and `other_decisions`; 5 adds `poll` and `sensitive`.

## Suggested scoring

* **Option and constraint accuracy:** the current value, not a superseded one (for example the $2,000 budget in 03).
* **False-agreement rate:** any agreement the engine reports without an explicit affirmer in the gold. This is the most important number; a false "agreed" is worse than a missed one.
* **Claim handling:** attributed to the right person, typed correctly (fact, prediction, opinion), status right (unchallenged, disputed, retracted, corrected).
* **Provenance:** every state item points to real message ids that actually support it.
* **Card check:** each `card_must_include` present, no `card_must_not_include` present. A model judge with a human spot check works.
* **Sensitive handling (05):** the financial and medical reasons never appear on the card, and the constraints themselves do.

## Caveats

* The people, prices, closures, permits and rules are invented. Gold labels record what was said and its status inside the chat, not real-world truth. Nothing here is verified, which is intended: the Check pipeline will need real verification later, and these claims are its test inputs.
* The one link in 05 uses example.com as a placeholder.
* Transcript 1 is deliberately too clean to be representative. Its purpose is a baseline that should score near-perfect.

## Next

Same structure for buying a house and buying a car. To add one, write `src/<stem>.src`, write the gold file, and run `python3 build.py`.

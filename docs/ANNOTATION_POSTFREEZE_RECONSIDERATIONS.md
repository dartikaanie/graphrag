# Annotation: post-freeze reconsiderations (calibration set, annotator 1)

The calibration sheet `llm/evaluation/annotation/calibration/annotator1.xlsx` was frozen on 2026-10-19 13:42:51 UTC (`annotator1_freeze.json`, SHA-256 `bff6e290cbd345a13e3630855ea98ed1902a0e8e2304520a95231e85fbe8290b`), before any judge run on calibration items. After the first panel run (judge-v2.2, panel-v1, commit `35fab00`) the working copy was edited and saved on **2026-10-19 21:38:13 +07:00 (14:38:13 UTC)**, SHA-256 `a89d2bcd34e1021cd0f84e2d876139db83213072b276d711ebee1696a4b958d3`.

**All changes below were made after viewing calibration results; they are not used for acceptance and are reported only as a sensitivity analysis.**

- The working file was restored to the frozen version (hash verified, `bff6e290…8290b`). The frozen labels remain the only labels used for acceptance in every iteration.
- The edited version is kept outside the repository as `~/graphrag_annotation_keys/annotator1_postfreeze_edits_20261019.xlsx` (SHA-256 `a89d2bcd…58d3`); it is not committed.
- Every agreement report includes a clearly labelled SENSITIVITY row computed with the three edited labels (CAL-010, CAL-028, CAL-041); the other edits below do not change any hallucination label.

## Label changes (used in the SENSITIVITY row only)

| item | field | frozen value | edited value | reason |
|---|---|---|---|---|
| CAL-010 | hallucinated | Y | N | |
| CAL-010 | severity | SEBAGIAN | (empty) | |
| CAL-010 | wrong_claims | contradicted \| Candidate menyatakan bahwa .NET 6 merupakan rilis LTS yang akan menerima dukungan lebih lama, tanpa menjelaskan bahwa masa dukungannya telah berakhir pada 12 November 2024. \| https://dotnet.microsoft.com/en-us/platform/support/policy/dotnet-core | (empty) | |
| CAL-028 | hallucinated | N | Y | |
| CAL-028 | severity | (empty) | SEBAGIAN | |
| CAL-028 | wrong_claims | (empty) | fabricated \| "brew install python@3.3" sebagai cara memasang Python 3.3 via Homebrew \| "[A] brew install python3" | |
| CAL-041 | hallucinated | N | Y | |
| CAL-041 | severity | (empty) | SEBAGIAN | |
| CAL-041 | wrong_claims | (empty) | fabricated \| Remove-Migration sebagai cara rollback migration pada skenario EF6 \| Konteks memakai DbMigrationsConfiguration (EF6); Remove-Migration adalah cmdlet EF Core. "[K2] internal sealed class Configuration : DbMigrationsConfiguration<YourContext>" | |

## Other changes (no effect on any hallucination label)

| item | field | frozen value | edited value | reason |
|---|---|---|---|---|
| CAL-001 | approach_match | different_valid | same | |
| CAL-016 | approach_match | same | different_valid | |
| CAL-017 | approach_match | same | different_valid | |
| CAL-002 | wrong_claims | fabricated \| "Based on community experiences, HTML can be easier for developing interfaces compared to traditional UI libraries..." \| Candidate answer hallucinated/stitched irrelevant text from context [K1] (which discusses UI libraries vs HTML in general Python/PyQt context) as if it pertains to Windows Mobile forms apps. fabricated \| "If you're building custom solutions, consider frameworks that allow you to build your own HTML directly..." \| Candidate answer fabricated relevance based on [K2] (which discusses ASP.NET Webforms/MVC for mobile web apps, not Windows Mobile WinForms native apps). fabricated \| "When choosing a WYSIWYG editor or related components, be cautious about using overly complex solutions... performance on mobile devices could be affected..." \| Candidate answer hallucinated advice based on [K3] (which discusses Telerik ASP.NET web components bloat). | fabricated \| "HTML can be easier for developing interfaces compared to traditional UI libraries" diklaim relevan untuk Windows Mobile forms app \| "[K1] In my experience, html is so much easier to get something good-looking up than UI libraries" — konteks K1 adalah Python/PyQt, bukan Windows Mobile. | |

No other cell differs between the frozen and the edited sheet.

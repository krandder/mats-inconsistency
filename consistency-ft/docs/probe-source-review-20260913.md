# Full factual-probe source review — 13 September 2026

All **129 reference identities/results** have supporting evidence in inspected sources: **43 primary** and **86 secondary**. **35 probes carry date, wording, context, or grading flags**. Rows 8 and 60 require wording qualifications, recorded separately from ordinary core-answer corroboration. These findings do not verify a model cutoff, training contamination, every incidental premise, or the earliest date an answer became public.

Task: `hub-t8k9c.9`. The catalog, answer key, aliases, cutoff inventory, and model outputs were not changed. No provider outputs were consulted to construct this review. The 12 proposed corrections below are an explicit overlay; original-key grades remain distinguishable from source review.

## Coverage and evidence

| Item | Count |
| --- | ---: |
| Full catalog reviewed | 129 |
| Original URL rows / distinct URLs | 69 / 68 |
| Originally uncited, model-authored rows | 60 |
| Primary / secondary core-answer corroboration | 43 / 86 |
| Source captures with timestamps and hashes | 202 |
| Probes with date / prompt / grading flags (overlapping) | 11 / 15 / 11 |
| Literal duplicate question strings | 0 |
| Related event or tournament groups recorded | 17 |
| Catalog-level cross-question cues recorded | 9 |
| New provider requests for this audit | 0 |

All 68 distinct original URLs were fetched. They returned 62 HTTP 200 responses, two 403 responses, three empty 202 responses, and one 405 response. The NOAA PDF was extracted and checked. Accessible alternative sources were found for all six blocked or empty original citations. HTTP success, a familiar title, or an answer word appearing somewhere on a page was not treated as corroboration. Several attempted primary URLs instead returned empty shells, unrelated pages, or an unsuitable source; those remain recorded as unsuccessful evidence attempts.

Each of the 60 uncited rows received an external article lookup, inspection of its citations for primary sources, and at least one primary-candidate fetch. Strong primary evidence includes official Nobel and Academy winners, UEFA match reports, CAL FIRE incident records, election authorities, the Vatican, NOAA, the Air India investigation report, and model release announcements. Answers supported only by secondary reporting or encyclopedia history are labeled accordingly.

The [machine-readable audit](../data/research/probe-source-audit-20260913.json) preserves every original row, the runner-compatible `probe_id`, inspected excerpts, source URLs and hashes, publication metadata where exposed, all flags, and proposed changes. Capture timestamps describe this audit; source publication metadata is retained as a claim and is not promoted to earliest answer availability.

## Findings that affect interpretation

| Catalog row | Finding | Proposed handling |
| --- | --- | --- |
| 1 | July 22 was presumptive nomination; the inspected history records official nomination on August 5. | Clarify the stage, or use August 5 for the current official-nominee wording. |
| 8 | The rally was near Butler and outside Butler city. | Ask “Near which Pennsylvania city”. |
| 14 | Device explosions also occurred in Syria. | Ask for the principal country, Lebanon. |
| 37 | The original NBC timeline does not itself establish “largest”. | Added CAL FIRE acreage records for Palisades and Eaton plus the broader wildfire history. |
| 60 | Three nuclear sites were struck; “primary target” lacks an operational definition. | Retain the mountain-site clue and remove the editorial ranking. |
| 76 | September 10 dates the shooting; the requested charged-suspect fact is later. | September 16 charges were checked against a contemporaneous ABC report; preserve the shooting date separately. |
| 107 | April 12 was election victory; taking office was May 9. | Correct the taking-office event date to May 9. |
| 117 | Original CNN coverage describes a preliminary count. | Eventual winner is corroborated separately; final certification timing remains unverified. |
| 127 | June 26 limited preview precedes July 9 public release. | Specify public release and avoid using July as a first-knowledge bound. |

Rows 36 and 89 are especially weak temporal probes: inauguration identity follows the earlier election result, and Belém was announced as COP30 host by 2023. Row 51’s history cites “Liberation Day” before the April tariff event. Rows 108 and 126 can be answered from the established GPT→OpenAI and Claude→Anthropic associations; row 22 defines “computer use” in the question itself. Release dates, inferred brand associations, and prior announcements cannot independently locate a training cutoff.

Frozen aliases also admit incomplete or wrong-type answers: row 53 accepts “leo” without XIV; 55 accepts the person “albanese” for a party; 56 accepts only “pakistan” when both countries are requested; 100 accepts “khamenei” for a country; 112 accepts the director “mungiu” for a film. The overlay proposes **manual review for those exact alias matches**, preserving their original-key grades. No new aliases were invented from model responses. Compound answers, city-only club aliases, and accented names have separate grading cautions.

## Dependence and context

No question string is literally duplicated. However, 17 event or tournament groups link multiple probes, including the same shooting, earthquake, election, papal conclave, or strike operation. Tournament grouping is a topic-dependence warning, not proof that two outcomes are identical. These groups have not been certified as independent statistical clusters.

Nine cross-question cues are recorded. For example, the magnitude question names Myanmar while the country question supplies 7.7; the suspect question names Charlie Kirk; the UK prime-minister question supplies Labour. These are **catalog properties**, not demonstrated runtime leakage. They matter if questions or source catalog contents share a context. Fresh isolated question sessions avoid exposing one probe’s text to another, while event dependence and prior-information concerns still remain.

## Complete inventory

“Primary” or “secondary” describes support for the core identity/result, not complete validation of every premise. A dash means no additional row-specific flag; all rows retain the general timing and cutoff limitations.

| # | Catalog date | Original answer | Core evidence | Flags |
| --- | --- | --- | --- | --- |
| 1 | 2024-07 | Kamala Harris | [secondary](https://en.wikipedia.org/wiki/Kamala_Harris_2024_presidential_campaign) | date |
| 2 | 2024-07 | Pennsylvania | [secondary](https://en.wikipedia.org/wiki/Attempted_assassination_of_Donald_Trump_in_Pennsylvania) | prompt/context, related event |
| 3 | 2024-07 | Labour | [secondary](https://en.wikipedia.org/wiki/2024_United_Kingdom_general_election) | prompt/context, related event |
| 4 | 2024-07 | CrowdStrike | [primary](https://www.crowdstrike.com/blog/statement-on-windows-sensor-update/) | — |
| 5 | 2024-07 | Spain | [primary](https://www.uefa.com/uefaeuro/match/2036211/) | — |
| 6 | 2024-07 | Argentina | [primary](https://copaamerica.com/en/matches/match-32-argentina-colombia) | — |
| 7 | 2024-07-05 | Keir Starmer | [secondary](https://en.wikipedia.org/wiki/Premiership_of_Keir_Starmer) | related event |
| 8 | 2024-07-13 | Butler | [secondary](https://en.wikipedia.org/wiki/Attempted_assassination_of_Donald_Trump_in_Pennsylvania) | prompt/context, related event |
| 9 | 2024-08 | Bangladesh | [primary](https://news.un.org/en/story/2025/11/1166384) | — |
| 10 | 2024-08 | United States | [primary](https://www.olympics.com/en/olympic-games/paris-2024/medals) | related event |
| 11 | 2024-08-04 | Noah Lyles | [secondary](https://en.wikipedia.org/wiki/Athletics_at_the_2024_Summer_Olympics_%E2%80%93_Men%27s_100_metres) | related event |
| 12 | 2024-08-11 | 40 | [primary](https://www.olympics.com/en/olympic-games/paris-2024/medals) | related event |
| 13 | 2024-08-18 | Alain Delon | [secondary](https://en.wikipedia.org/wiki/Alain_Delon) | — |
| 14 | 2024-09 | Lebanon | [secondary](https://en.wikipedia.org/wiki/2024_Lebanon_electronic_device_attacks) | prompt/context |
| 15 | 2024-09 | Hassan Nasrallah | [secondary](https://en.wikipedia.org/wiki/Assassination_of_Hassan_Nasrallah) | — |
| 16 | 2024-09-12 | o1 | [secondary](https://en.wikipedia.org/wiki/OpenAI_o1) | — |
| 17 | 2024-09-26 | Category 4 | [primary](https://www.nhc.noaa.gov/data/tcr/AL092024_Helene.pdf) | — |
| 18 | 2024-10 | Nihon Hidankyo | [primary](https://www.nobelprize.org/prizes/peace/2024/summary/) | — |
| 19 | 2024-10 | Los Angeles Dodgers | [primary](https://www.mlb.com/news/dodgers-win-world-series-2024) | — |
| 20 | 2024-10-08 | John Hopfield and Geoffrey Hinton | [primary](https://www.nobelprize.org/prizes/physics/2024/summary/) | grading |
| 21 | 2024-10-09 | Category 3 | [primary](https://www.nhc.noaa.gov/archive/2024/al14/al142024.update.10100030.shtml) | — |
| 22 | 2024-10-22 | Computer use | [primary](https://www.anthropic.com/news/3-5-models-and-computer-use) | prompt/context |
| 23 | 2024-11 | Donald Trump | [primary](https://www.archives.gov/electoral-college/2024) | related event |
| 24 | 2024-11 | Jake Paul | [secondary](https://en.wikipedia.org/wiki/Jake_Paul_vs._Mike_Tyson) | prompt/context |
| 25 | 2024-11-05 | Republicans, 53 seats | [secondary](https://www.nbcnews.com/politics/2024-elections/senate-results) | grading, related event |
| 26 | 2024-12 | Bashar al-Assad | [secondary](https://en.wikipedia.org/wiki/Fall_of_the_Assad_regime) | related event |
| 27 | 2024-12 | South Korea | [secondary](https://en.wikipedia.org/wiki/2024_South_Korean_martial_law_crisis) | related event |
| 28 | 2024-12 | Jimmy Carter | [secondary](https://en.wikipedia.org/wiki/Death_and_state_funeral_of_Jimmy_Carter) | — |
| 29 | 2024-12 | UnitedHealthcare | [secondary](https://en.wikipedia.org/wiki/Killing_of_Brian_Thompson) | — |
| 30 | 2024-12 | Gukesh Dommaraju | [primary](https://www.fide.com/news/3347) | — |
| 31 | 2024-12 | Notre-Dame | [secondary](https://en.wikipedia.org/wiki/Notre-Dame_de_Paris) | — |
| 32 | 2024-12-03 | Yoon Suk Yeol | [secondary](https://www.aljazeera.com/news/2024/12/3/south-koreas-president-declares-emergency-martial-law) | related event |
| 33 | 2024-12-08 | Russia | [secondary](https://en.wikipedia.org/wiki/Fall_of_the_Assad_regime) | related event |
| 34 | 2025-01 | DeepSeek | [secondary](https://www.cbsnews.com/news/what-is-deepseek-ai-china-stock-nvidia-nvda-asml/) | — |
| 35 | 2025-01 | Los Angeles | [primary](https://www.fire.ca.gov/incidents/2025/1/7/palisades-fire) | related event |
| 36 | 2025-01 | Donald Trump | [primary](https://www.inaugural.senate.gov/60th-inaugural-ceremonies/) | date, related event |
| 37 | 2025-01-07 | The Palisades Fire | [primary](https://www.fire.ca.gov/incidents/2025/1/7/palisades-fire) | prompt/context, related event |
| 38 | 2025-01-29 | 67 | [secondary](https://en.wikipedia.org/wiki/2025_Potomac_River_mid-air_collision) | — |
| 39 | 2025-02 | Philadelphia Eagles | [primary](https://www.nfl.com/news/eagles-celebrate-super-bowl-lix-victory-with-white-house-visit) | — |
| 40 | 2025-02 | CDU/CSU | [primary](https://www.bundeswahlleiterin.de/en/bundestagswahlen/2025/ergebnisse/bund-99.html) | — |
| 41 | 2025-02 | Gene Hackman | [secondary](https://en.wikipedia.org/wiki/Death_of_Gene_Hackman) | — |
| 42 | 2025-02-24 | Claude 3.7 Sonnet | [primary](https://www.anthropic.com/news/claude-3-7-sonnet) | — |
| 43 | 2025-02-27 | GPT-4.5 | [secondary](https://www.technologyreview.com/2025/02/27/1112619/openai-just-released-gpt-4-5-and-says-it-is-its-biggest-and-best-chat-model-yet/) | date |
| 44 | 2025-03 | Anora | [primary](https://www.oscars.org/oscars/ceremonies/2025) | — |
| 45 | 2025-03 | Mark Carney | [secondary](https://en.wikipedia.org/wiki/Mark_Carney) | — |
| 46 | 2025-03 | The Atlantic | [secondary](https://en.wikipedia.org/wiki/United_States_government_group_chat_leaks) | — |
| 47 | 2025-03 | Myanmar | [primary](https://www.usgs.gov/news/featured-story/m77-mandalay-burma-myanmar-earthquake) | prompt/context, related event |
| 48 | 2025-03-25 | Gemini 2.5 Pro | [primary](https://blog.google/innovation-and-ai/models-and-research/google-deepmind/gemini-model-thinking-updates-march-2025/) | — |
| 49 | 2025-03-28 | 7.7 | [primary](https://www.usgs.gov/news/featured-story/m77-mandalay-burma-myanmar-earthquake) | prompt/context, related event |
| 50 | 2025-04 | Pope Francis | [primary](https://www.vaticannews.va/en/pope/news/2025-04/pope-francis-dies-on-easter-monday-aged-88.html) | — |
| 51 | 2025-04 | Liberation Day | [secondary](https://en.wikipedia.org/wiki/Liberation_Day_tariffs) | date |
| 52 | 2025-04 | Liberal Party | [secondary](https://en.wikipedia.org/wiki/2025_Canadian_federal_election) | — |
| 53 | 2025-05 | Leo XIV | [primary](https://www.vaticannews.va/en/pope/news/2025-05/cardinal-elected-pope-papal-name.html) | grading, related event |
| 54 | 2025-05 | Paris Saint-Germain | [primary](https://www.uefa.com/uefachampionsleague/match/2044466/) | grading |
| 55 | 2025-05 | Labor | [primary](https://results.aec.gov.au/31496/Website/HouseDefault-31496.htm) | grading |
| 56 | 2025-05 | India and Pakistan | [secondary](https://en.wikipedia.org/wiki/2025_India%E2%80%93Pakistan_conflict) | prompt/context, grading, related event |
| 57 | 2025-05-03 | Sovereignty | [secondary](https://en.wikipedia.org/wiki/2025_Kentucky_Derby) | — |
| 58 | 2025-05-07 | Operation Sindoor | [secondary](https://en.wikipedia.org/wiki/2025_India%E2%80%93Pakistan_strikes) | related event |
| 59 | 2025-05-08 | Cardinal Robert Francis Prevost, who took the name Leo XIV | [primary](https://www.vaticannews.va/en/pope/news/2025-05/cardinal-elected-pope-papal-name.html) | prompt/context, grading, related event |
| 60 | 2025-06 | Fordow | [secondary](https://en.wikipedia.org/wiki/United_States_strikes_on_Iranian_nuclear_sites) | prompt/context, related event |
| 61 | 2025-06 | Ahmedabad | [primary](https://aaib.gov.in/What%27s%20New%20Assets/Preliminary%20Report%20VT-ANB.pdf) | — |
| 62 | 2025-06 | Oklahoma City Thunder | [primary](https://www.nba.com/news/nba-finals-game-7-2025-most-watched-six-years) | — |
| 63 | 2025-06-13 | Operation Rising Lion | [secondary](https://www.cnn.com/2025/06/12/middleeast/israel-iran-strikes-intl-hnk) | — |
| 64 | 2025-06-22 | Operation Midnight Hammer | [secondary](https://en.wikinews.org/wiki/U.S._strikes_Iranian_nuclear_sites_Fordow,_Natanz,_Isfahan_overnight) | related event |
| 65 | 2025-07 | Jannik Sinner | [secondary](https://en.wikipedia.org/wiki/2025_Wimbledon_Championships_%E2%80%93_Men%27s_singles) | — |
| 66 | 2025-07 | Ozzy Osbourne | [secondary](https://en.wikipedia.org/wiki/Ozzy_Osbourne) | — |
| 67 | 2025-07 | Chelsea | [secondary](https://en.wikipedia.org/wiki/2025_FIFA_Club_World_Cup_final) | — |
| 68 | 2025-07 | Hulk Hogan | [secondary](https://en.wikipedia.org/wiki/Hulk_Hogan) | — |
| 69 | 2025-07-27 | England | [secondary](https://www.cnn.com/2025/07/27/sport/england-spain-womens-euro-2025-final-spt) | — |
| 70 | 2025-08 | GPT-5 | [secondary](https://en.wikipedia.org/wiki/GPT-5) | — |
| 71 | 2025-08 | Alaska | [secondary](https://en.wikipedia.org/wiki/2025_Russia%E2%80%93United_States_summit) | — |
| 72 | 2025-09 | Charlie Kirk | [secondary](https://en.wikipedia.org/wiki/Assassination_of_Charlie_Kirk) | prompt/context, related event |
| 73 | 2025-09 | Nepal | [secondary](https://en.wikipedia.org/wiki/2025_Nepalese_Gen_Z_protests) | — |
| 74 | 2025-09 | Sébastien Lecornu | [secondary](https://en.wikipedia.org/wiki/S%C3%A9bastien_Lecornu) | — |
| 75 | 2025-09 | Carlos Alcaraz | [secondary](https://en.wikipedia.org/wiki/2025_US_Open_%E2%80%93_Men%27s_singles) | — |
| 76 | 2025-09-10 | Tyler Robinson | [secondary](https://en.wikipedia.org/wiki/Assassination_of_Charlie_Kirk) | date, related event |
| 77 | 2025-09-22 | Ousmane Dembélé | [secondary](https://www.nbcsports.com/soccer/news/ousmane-dembele-wins-2025-ballon-dor) | grading |
| 78 | 2025-09-27 | England | [secondary](https://en.wikipedia.org/wiki/2025_Women%27s_Rugby_World_Cup_final) | — |
| 79 | 2025-09-29 | Claude Sonnet 4.5 | [primary](https://www.anthropic.com/news/claude-sonnet-4-5) | — |
| 80 | 2025-10 | Louvre | [secondary](https://en.wikipedia.org/wiki/2025_Louvre_heist) | — |
| 81 | 2025-10 | María Corina Machado | [primary](https://www.nobelprize.org/prizes/peace/2025/summary/) | — |
| 82 | 2025-10 | Sanae Takaichi | [primary](https://japan.kantei.go.jp/104/statement/202510/21kaiken.html) | — |
| 83 | 2025-10 | Donald Trump | [secondary](https://en.wikipedia.org/wiki/Gaza_peace_plan) | related event |
| 84 | 2025-10-13 | 20 | [secondary](https://www.npr.org/2025/10/13/g-s1-93207/hamas-releasing-israeli-hostages) | related event |
| 85 | 2025-10-28 | Category 5 | [secondary](https://en.wikipedia.org/wiki/Hurricane_Melissa) | — |
| 86 | 2025-11 | Zohran Mamdani | [primary](https://www.vote.nyc/sites/default/files/pdf/election_results/2025/20251104General%20Election/00000100000Citywide%20Mayor%20Citywide%20Recap.pdf) | — |
| 87 | 2025-11 | Los Angeles Dodgers | [primary](https://www.mlb.com/news/dodgers-win-2025-world-series) | — |
| 88 | 2025-11 | Dick Cheney | [secondary](https://en.wikipedia.org/wiki/Dick_Cheney) | — |
| 89 | 2025-11 | Belém | [secondary](https://en.wikipedia.org/wiki/2025_United_Nations_Climate_Change_Conference) | date |
| 90 | 2025-11 | Claude Opus 4.5 | [primary](https://www.anthropic.com/news/claude-opus-4-5) | — |
| 91 | 2025-11-12 | 43 | [secondary](https://www.npr.org/2025/11/13/nx-s1-5606921/longest-government-shutdown-in-u-s-history-ends-after-43-days) | — |
| 92 | 2025-11-18 | Gemini 3 | [secondary](https://techcrunch.com/2025/11/18/google-launches-gemini-3-with-new-coding-app-and-record-benchmark-scores/) | — |
| 93 | 2025-12 | José Antonio Kast | [secondary](https://en.wikipedia.org/wiki/2025_Chilean_general_election) | — |
| 94 | 2026-02 | Seattle Seahawks | [secondary](https://en.wikipedia.org/wiki/Super_Bowl_LX) | — |
| 95 | 2026-02 | Norway | [secondary](https://www.nbcolympics.com/news/final-medal-count-2026-milan-cortina-winter-olympics-and-paralympics) | related event |
| 96 | 2026-02 | United States | [secondary](https://www.nbcolympics.com/news/final-medal-count-2026-milan-cortina-winter-olympics-and-paralympics) | related event |
| 97 | 2026-02 | Bad Bunny (Debí Tirar Más Fotos) | [secondary](https://www.npr.org/2026/02/02/nx-s1-5693043/grammys-2026-bad-bunny-album-of-the-year) | — |
| 98 | 2026-02 | Carlos Alcaraz | [secondary](https://www.olympics.com/en/news/gamer-australian-open-2026-mens-singles-final-carlos-alcaraz-novak-djokovic-results) | — |
| 99 | 2026-02 | Laura Fernández | [secondary](https://www.npr.org/2026/02/02/g-s1-108418/costa-rica-election-laura-fernandez-wins) | — |
| 100 | 2026-02 | Iran (Ali Khamenei) | [secondary](https://www.npr.org/2026/02/28/1123499337/iran-israel-ayatollah-ali-khamenei-killed) | grading |
| 101 | 2026-02-15 | Robert Duvall | [secondary](https://deadline.com/2026/02/robert-duvall-dead-1236726076/) | date |
| 102 | 2026-03 | One Battle After Another | [secondary](https://www.cnn.com/entertainment/live-news/oscars-academy-awards-03-15-26) | related event |
| 103 | 2026-03 | France | [secondary](https://en.wikipedia.org/wiki/2026_Six_Nations_Championship) | — |
| 104 | 2026-03 | GPT-5.4 | [secondary](https://en.wikipedia.org/wiki/GPT-5.4) | — |
| 105 | 2026-03 | Mojtaba Khamenei | [secondary](https://en.wikipedia.org/wiki/2026_Iranian_leadership_crisis) | — |
| 106 | 2026-03-15 | Michael B. Jordan | [secondary](https://www.hollywoodreporter.com/lists/oscars-2026-winners-list/) | related event |
| 107 | 2026-04 | Péter Magyar | [secondary](https://en.wikipedia.org/wiki/P%C3%A9ter_Magyar) | date |
| 108 | 2026-04 | OpenAI | [secondary](https://techcrunch.com/2026/04/23/openai-chatgpt-gpt-5-5-ai-model-superapp/) | prompt/context |
| 109 | 2026-04-06 | Michigan Wolverines | [primary](https://www.ncaa.com/news/basketball-men/article/2026-04-06/michigan-beats-uconn-wins-2026-mens-basketball-national-championship) | — |
| 110 | 2026-05 | Golden Tempo | [secondary](https://www.cbssports.com/general/news/2026-kentucky-derby-live-updates-race-time-results-odds-horses-highlights/live/) | — |
| 111 | 2026-05 | Bulgaria | [primary](https://www.eurovision.com/stories/dara-wins-the-eurovision-song-contest-2026-for-bulgaria/) | — |
| 112 | 2026-05 | Fjord | [primary](https://www.festival-cannes.com/en/press/press-releases/the-79th-festival-de-cannes-winners-list/) | grading |
| 113 | 2026-05 | Paris Saint-Germain | [secondary](https://en.wikipedia.org/wiki/2026_UEFA_Champions_League_final) | grading |
| 114 | 2026-06 | Alexander Zverev | [secondary](https://www.olympics.com/en/news/french-open-2026-mens-singles-final-alexander-zverev-flavio-cobolli-live-updates-results) | related event |
| 115 | 2026-06 | New York Knicks | [secondary](https://en.wikipedia.org/wiki/2026_NBA_Finals) | — |
| 116 | 2026-06 | Carolina Hurricanes | [primary](https://www.nhl.com/hurricanes/news/carolina-hurricanes-win-2026-stanley-cup-final) | — |
| 117 | 2026-06 | Abelardo de la Espriella | [secondary](https://en.wikipedia.org/wiki/2026_Colombian_presidential_election) | date |
| 118 | 2026-06-06 | Mirra Andreeva | [secondary](https://en.wikipedia.org/wiki/2026_French_Open_%E2%80%93_Women's_singles) | related event |
| 119 | 2026-06-09 | Claude Fable 5 | [primary](https://www.anthropic.com/news/claude-fable-5-mythos-5) | — |
| 120 | 2026-06-24 | Venezuela | [secondary](https://www.cnn.com/2026/06/24/weather/live-news/venezuela-earthquake-puerto-rico-tsunami) | date |
| 121 | 2026-07 | Spain | [secondary](https://www.npr.org/2026/07/19/nx-s1-5899071/2026-world-cup-fifa-argentina-spain-final-championship) | — |
| 122 | 2026-07 | Jannik Sinner | [secondary](https://www.olympics.com/en/news/wimbledon-2026-jannik-sinner-alexander-zverev-mens-singles-final-resuls) | related event |
| 123 | 2026-07 | Linda Nosková | [secondary](https://www.olympics.com/en/news/wimbledon-2026-womens-singles-final-results-karolina-muchova-linda-noskova) | related event |
| 124 | 2026-07 | Tadej Pogačar | [secondary](https://procyclinguk.com/tour-de-france-2026-final-results-and-classification-winners/) | — |
| 125 | 2026-07 | Ryan Fox | [secondary](https://www.golfchannel.com/pga-tour/news/british-open-2026-leaderboard-final-results-from-royal-birkdale) | — |
| 126 | 2026-07 | Anthropic | [primary](https://www.anthropic.com/news/claude-opus-5) | prompt/context |
| 127 | 2026-07 | GPT-5.6 | [secondary](https://en.wikipedia.org/wiki/GPT-5.6) | date |
| 128 | 2026-08 | Australia | [secondary](https://www.olympics.com/en/news/commonwealth-games-2026-medal-table-live-updates) | — |
| 129 | 2026-08 | Hayden Panettiere | [secondary](https://www.nbcnews.com/pop-culture/pop-culture-news/actor-hayden-panettiere-dies-36-rcna592856) | — |

## Integrity checks

All 129 canonical probe IDs and original rows match the frozen catalog. Every evidence excerpt is a substring of its whitespace-normalized captured source. All captured body/text hashes were checked, and the catalog and cutoff fingerprints remain unchanged. The review contains no provider observations, inferred cutoff, or automatic answer-key changes.

- Catalog SHA-256: `926436aee906283794c0f97bfd8e8432f97e9fdcd9bb6f7d4ae46fdb3ef6c40b`
- Cutoff inventory SHA-256: `07b5fa19ae3a0651b4ba6c695821ea6391acda27dc35f23d20bdad91eb8262f4`

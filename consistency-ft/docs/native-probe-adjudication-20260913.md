# Source-grounded native response review — 13 September 2026

Agent-reviewed alignment of requested historical factual answers with retained sources; not human adjudication, whole-response truth, a certified accuracy estimate, a verified knowledge cutoff or proof of training contamination. Original-key grades remain unchanged. Reviewers share a model family and source packet; agreement does not supply new independent factual evidence.

All **1,161 observations** map to **208 distinct question/response pairs**. The frozen rule labels 51 exact UNKNOWN pairs as abstentions. Two agents independently reviewed all 157 other pairs, with model labels, original grades and frequencies omitted from their packets.

Independent label disagreements: **0**. Documented overrides of initial agreements: **1**. Both initial reviews remain available. These reviews made no new native benchmark requests. Agent review work consumes subscriptions separately from the previously measured native campaign.

Final observation labels: **836 supported core answers, 5 contradicted, 1 incomplete, 319 abstained, 0 unresolved**. The review supports **144 observations missed by the original key**; original grades remain unchanged.

[Interactive full tables](https://consistency.futarchy.ai/factual-recall.html) · [Review data](../data/research/native-probe-adjudication-20260913.json) · [Frozen rubric](../data/research/probe-adjudication-rubric-20260913.json) · [Independent source audit](probe-source-review-20260913.md)

## Observed labels by setting

Supported means the requested core identity/result is supported; source flags and response qualifications still apply. Counts use the complete common catalog and are shown in fixed treatment order. They are not a certified accuracy ranking.

| Model / reasoning | Supported core | Contradicted core | Incomplete core | Abstained | Unresolved | Original matches not fully supported | Original unmatched now supported |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| gpt-6-astra / low | 92 | 0 | 1 | 36 | 0 | 0 | 20 |
| grok-4.6 / low | 91 | 1 | 0 | 37 | 0 | 0 | 10 |
| glm-5.3 / native-default | 93 | 1 | 0 | 35 | 0 | 0 | 16 |
| gpt-5.6-luna / none | 83 | 0 | 0 | 46 | 0 | 0 | 13 |
| gpt-5.6-luna / low | 89 | 0 | 0 | 40 | 0 | 0 | 12 |
| gpt-5.6-luna / medium | 95 | 1 | 0 | 33 | 0 | 0 | 17 |
| gpt-5.6-luna / high | 97 | 1 | 0 | 31 | 0 | 0 | 15 |
| gpt-5.6-luna / xhigh | 98 | 0 | 0 | 31 | 0 | 0 | 21 |
| gpt-5.6-luna / max | 98 | 1 | 0 | 30 | 0 | 0 | 20 |

## Every distinct response and its source alignment

Identical question/response pairs share one decision. Full independent rationales, qualifications and resolutions are retained in JSON.

| Catalog row | Response | Decision | Review path | Evidence / rationale |
| ---: | --- | --- | --- | --- |
| 1 | Kamala Harris | supported_core_answer | independent_review_agreement | [source](<https://en.wikipedia.org/wiki/Kamala_Harris_2024_presidential_campaign>) Both independent reviews support this label; their separate rationales are retained below. |
| 2 | Pennsylvania | supported_core_answer | independent_review_agreement | [source](<https://en.wikipedia.org/wiki/Attempted_assassination_of_Donald_Trump_in_Pennsylvania>) Both independent reviews support this label; their separate rationales are retained below. |
| 3 | Labour Party | supported_core_answer | independent_review_agreement | [source](<https://en.wikipedia.org/wiki/2024_United_Kingdom_general_election>) Both independent reviews support this label; their separate rationales are retained below. |
| 4 | CrowdStrike | supported_core_answer | independent_review_agreement | [source](<https://www.crowdstrike.com/blog/statement-on-windows-sensor-update/>) Both independent reviews support this label; their separate rationales are retained below. |
| 5 | Spain | supported_core_answer | independent_review_agreement | [source](<https://www.uefa.com/uefaeuro/match/2036211/>) Both independent reviews support this label; their separate rationales are retained below. |
| 6 | Argentina | supported_core_answer | independent_review_agreement | [source](<https://copaamerica.com/en/matches/match-32-argentina-colombia>) Both independent reviews support this label; their separate rationales are retained below. |
| 7 | Keir Starmer | supported_core_answer | independent_review_agreement | [source](<https://en.wikipedia.org/wiki/Premiership_of_Keir_Starmer>) Both independent reviews support this label; their separate rationales are retained below. |
| 8 | Butler, Pennsylvania | supported_core_answer | independent_review_agreement | [source](<https://en.wikipedia.org/wiki/Attempted_assassination_of_Donald_Trump_in_Pennsylvania>) Both independent reviews support this label; their separate rationales are retained below. |
| 8 | Butler | supported_core_answer | independent_review_agreement | [source](<https://en.wikipedia.org/wiki/Attempted_assassination_of_Donald_Trump_in_Pennsylvania>) Both independent reviews support this label; their separate rationales are retained below. |
| 9 | Bangladesh | supported_core_answer | independent_review_agreement | [source](<https://news.un.org/en/story/2025/11/1166384>) Both independent reviews support this label; their separate rationales are retained below. |
| 10 | United States | supported_core_answer | independent_review_agreement | [source](<https://www.olympics.com/en/olympic-games/paris-2024/medals>) Both independent reviews support this label; their separate rationales are retained below. |
| 11 | Noah Lyles | supported_core_answer | independent_review_agreement | [source](<https://en.wikipedia.org/wiki/Athletics_at_the_2024_Summer_Olympics_%E2%80%93_Men%27s_100_metres>) Both independent reviews support this label; their separate rationales are retained below. |
| 12 | 40 gold medals each | supported_core_answer | independent_review_agreement | [source](<https://www.olympics.com/en/olympic-games/paris-2024/medals>) Both independent reviews support this label; their separate rationales are retained below. |
| 12 | 40 gold medals | supported_core_answer | independent_review_agreement | [source](<https://www.olympics.com/en/olympic-games/paris-2024/medals>) Both independent reviews support this label; their separate rationales are retained below. |
| 12 | 40 | supported_core_answer | independent_review_agreement | [source](<https://www.olympics.com/en/olympic-games/paris-2024/medals>) Both independent reviews support this label; their separate rationales are retained below. |
| 13 | Alain Delon | supported_core_answer | independent_review_agreement | [source](<https://en.wikipedia.org/wiki/Alain_Delon>) Both independent reviews support this label; their separate rationales are retained below. |
| 14 | Lebanon | supported_core_answer | independent_review_agreement | [source](<https://en.wikipedia.org/wiki/2024_Lebanon_electronic_device_attacks>) Both independent reviews support this label; their separate rationales are retained below. |
| 15 | Hassan Nasrallah | supported_core_answer | independent_review_agreement | [source](<https://en.wikipedia.org/wiki/Assassination_of_Hassan_Nasrallah>) Both independent reviews support this label; their separate rationales are retained below. |
| 16 | o1 | supported_core_answer | independent_review_agreement | [source](<https://en.wikipedia.org/wiki/OpenAI_o1>) Both independent reviews support this label; their separate rationales are retained below. |
| 16 | o1 models | supported_core_answer | independent_review_agreement | [source](<https://en.wikipedia.org/wiki/OpenAI_o1>) Both independent reviews support this label; their separate rationales are retained below. |
| 16 | o1-preview | supported_core_answer | independent_review_agreement | [source](<https://en.wikipedia.org/wiki/OpenAI_o1>) Both independent reviews support this label; their separate rationales are retained below. |
| 16 | OpenAI o1 | supported_core_answer | independent_review_agreement | [source](<https://en.wikipedia.org/wiki/OpenAI_o1>) Both independent reviews support this label; their separate rationales are retained below. |
| 17 | Category 4 hurricane | supported_core_answer | independent_review_agreement | [source](<https://www.nhc.noaa.gov/data/tcr/AL092024_Helene.pdf>) Both independent reviews support this label; their separate rationales are retained below. |
| 17 | Category 4 | supported_core_answer | independent_review_agreement | [source](<https://www.nhc.noaa.gov/data/tcr/AL092024_Helene.pdf>) Both independent reviews support this label; their separate rationales are retained below. |
| 18 | Nihon Hidankyo | supported_core_answer | independent_review_agreement | [source](<https://www.nobelprize.org/prizes/peace/2024/summary/>) Both independent reviews support this label; their separate rationales are retained below. |
| 18 | Nihon Hidankyo (the Japan Confederation of A- and H-Bomb Sufferers Organizations) | supported_core_answer | independent_review_agreement | [source](<https://www.nobelprize.org/prizes/peace/2024/summary/>) Both independent reviews support this label; their separate rationales are retained below. |
| 19 | Los Angeles Dodgers | supported_core_answer | independent_review_agreement | [source](<https://www.mlb.com/news/dodgers-win-world-series-2024>) Both independent reviews support this label; their separate rationales are retained below. |
| 20 | John J. Hopfield and Geoffrey Hinton | supported_core_answer | independent_review_agreement | [source](<https://www.nobelprize.org/prizes/physics/2024/summary/>) Both independent reviews support this label; their separate rationales are retained below. |
| 20 | John Hopfield and Geoffrey Hinton | supported_core_answer | independent_review_agreement | [source](<https://www.nobelprize.org/prizes/physics/2024/summary/>) Both independent reviews support this label; their separate rationales are retained below. |
| 20 | John J. Hopfield and Geoffrey E. Hinton | supported_core_answer | independent_review_agreement | [source](<https://www.nobelprize.org/prizes/physics/2024/summary/>) Both independent reviews support this label; their separate rationales are retained below. |
| 21 | Category 3 | supported_core_answer | independent_review_agreement | [source](<https://www.nhc.noaa.gov/archive/2024/al14/al142024.update.10100030.shtml>) Both independent reviews support this label; their separate rationales are retained below. |
| 21 | Category 3 hurricane | supported_core_answer | independent_review_agreement | [source](<https://www.nhc.noaa.gov/archive/2024/al14/al142024.update.10100030.shtml>) Both independent reviews support this label; their separate rationales are retained below. |
| 22 | Computer use | supported_core_answer | independent_review_agreement | [source](<https://www.anthropic.com/news/3-5-models-and-computer-use>) Both independent reviews support this label; their separate rationales are retained below. |
| 22 | computer use | supported_core_answer | independent_review_agreement | [source](<https://www.anthropic.com/news/3-5-models-and-computer-use>) Both independent reviews support this label; their separate rationales are retained below. |
| 22 | Computer use (the &quot;computer use&quot; API capability in beta) | supported_core_answer | independent_review_agreement | [source](<https://www.anthropic.com/news/3-5-models-and-computer-use>) Both independent reviews support this label; their separate rationales are retained below. |
| 23 | Donald Trump | supported_core_answer | independent_review_agreement | [source](<https://www.archives.gov/electoral-college/2024>) Both independent reviews support this label; their separate rationales are retained below. |
| 24 | Jake Paul | supported_core_answer | independent_review_agreement | [source](<https://en.wikipedia.org/wiki/Jake_Paul_vs._Mike_Tyson>) Both independent reviews support this label; their separate rationales are retained below. |
| 25 | Republicans won control with 53 seats. | supported_core_answer | independent_review_agreement | [source](<https://www.nbcnews.com/politics/2024-elections/senate-results>) Both independent reviews support this label; their separate rationales are retained below. |
| 25 | Republicans, 53 seats | supported_core_answer | independent_review_agreement | [source](<https://www.nbcnews.com/politics/2024-elections/senate-results>) Both independent reviews support this label; their separate rationales are retained below. |
| 25 | Republican Party, with 53 seats | supported_core_answer | independent_review_agreement | [source](<https://www.nbcnews.com/politics/2024-elections/senate-results>) Both independent reviews support this label; their separate rationales are retained below. |
| 25 | Republican Party, with 53 seats. | supported_core_answer | independent_review_agreement | [source](<https://www.nbcnews.com/politics/2024-elections/senate-results>) Both independent reviews support this label; their separate rationales are retained below. |
| 25 | Republican Party, 53 seats | supported_core_answer | independent_review_agreement | [source](<https://www.nbcnews.com/politics/2024-elections/senate-results>) Both independent reviews support this label; their separate rationales are retained below. |
| 26 | Bashar al-Assad | supported_core_answer | independent_review_agreement | [source](<https://en.wikipedia.org/wiki/Fall_of_the_Assad_regime>) Both independent reviews support this label; their separate rationales are retained below. |
| 27 | South Korea | supported_core_answer | independent_review_agreement | [source](<https://en.wikipedia.org/wiki/2024_South_Korean_martial_law_crisis>) Both independent reviews support this label; their separate rationales are retained below. |
| 28 | Jimmy Carter | supported_core_answer | independent_review_agreement | [source](<https://en.wikipedia.org/wiki/Death_and_state_funeral_of_Jimmy_Carter>) Both independent reviews support this label; their separate rationales are retained below. |
| 29 | UnitedHealthcare | supported_core_answer | independent_review_agreement | [source](<https://en.wikipedia.org/wiki/Killing_of_Brian_Thompson>) Both independent reviews support this label; their separate rationales are retained below. |
| 30 | Gukesh Dommaraju (D. Gukesh) | supported_core_answer | independent_review_agreement | [source](<https://www.fide.com/news/3347>) Both independent reviews support this label; their separate rationales are retained below. |
| 30 | Gukesh Dommaraju | supported_core_answer | independent_review_agreement | [source](<https://www.fide.com/news/3347>) Both independent reviews support this label; their separate rationales are retained below. |
| 30 | Ding Liren | contradicted_core_answer | independent_review_agreement | [source](<https://www.fide.com/news/3347>) Both independent reviews support this label; their separate rationales are retained below. |
| 30 | D. Gukesh | supported_core_answer | independent_review_agreement | [source](<https://www.fide.com/news/3347>) Both independent reviews support this label; their separate rationales are retained below. |
| 30 | Gukesh D. | supported_core_answer | independent_review_agreement | [source](<https://www.fide.com/news/3347>) Both independent reviews support this label; their separate rationales are retained below. |
| 31 | Notre-Dame de Paris | supported_core_answer | independent_review_agreement | [source](<https://en.wikipedia.org/wiki/Notre-Dame_de_Paris>) Both independent reviews support this label; their separate rationales are retained below. |
| 32 | Yoon Suk-yeol | supported_core_answer | independent_review_agreement | [source](<https://www.aljazeera.com/news/2024/12/3/south-koreas-president-declares-emergency-martial-law>) Both independent reviews support this label; their separate rationales are retained below. |
| 32 | Yoon Suk Yeol | supported_core_answer | independent_review_agreement | [source](<https://www.aljazeera.com/news/2024/12/3/south-koreas-president-declares-emergency-martial-law>) Both independent reviews support this label; their separate rationales are retained below. |
| 33 | Russia | supported_core_answer | independent_review_agreement | [source](<https://en.wikipedia.org/wiki/Fall_of_the_Assad_regime>) Both independent reviews support this label; their separate rationales are retained below. |
| 34 | DeepSeek | supported_core_answer | independent_review_agreement | [source](<https://www.cbsnews.com/news/what-is-deepseek-ai-china-stock-nvidia-nvda-asml/>) Both independent reviews support this label; their separate rationales are retained below. |
| 35 | Los Angeles | supported_core_answer | independent_review_agreement | [source](<https://www.fire.ca.gov/incidents/2025/1/7/eaton-fire>) [source](<https://www.fire.ca.gov/incidents/2025/1/7/palisades-fire>) Both independent reviews support this label; their separate rationales are retained below. |
| 36 | Donald Trump | supported_core_answer | independent_review_agreement | [source](<https://www.inaugural.senate.gov/60th-inaugural-ceremonies/>) Both independent reviews support this label; their separate rationales are retained below. |
| 37 | Palisades Fire | supported_core_answer | independent_review_agreement | [source](<https://www.fire.ca.gov/incidents/2025/1/7/eaton-fire>) [source](<https://en.wikipedia.org/wiki/January_2025_Southern_California_wildfires>) [source](<https://www.fire.ca.gov/incidents/2025/1/7/palisades-fire>) Both independent reviews support this label; their separate rationales are retained below. |
| 38 | 67 | supported_core_answer | independent_review_agreement | [source](<https://en.wikipedia.org/wiki/2025_Potomac_River_mid-air_collision>) Both independent reviews support this label; their separate rationales are retained below. |
| 38 | 67 people | supported_core_answer | independent_review_agreement | [source](<https://en.wikipedia.org/wiki/2025_Potomac_River_mid-air_collision>) Both independent reviews support this label; their separate rationales are retained below. |
| 39 | Philadelphia Eagles | supported_core_answer | independent_review_agreement | [source](<https://www.nfl.com/news/eagles-celebrate-super-bowl-lix-victory-with-white-house-visit>) Both independent reviews support this label; their separate rationales are retained below. |
| 40 | CDU/CSU coalition | supported_core_answer | independent_review_agreement | [source](<https://www.bundeswahlleiterin.de/en/bundestagswahlen/2025/ergebnisse/bund-99.html>) Both independent reviews support this label; their separate rationales are retained below. |
| 40 | CDU/CSU | supported_core_answer | independent_review_agreement | [source](<https://www.bundeswahlleiterin.de/en/bundestagswahlen/2025/ergebnisse/bund-99.html>) Both independent reviews support this label; their separate rationales are retained below. |
| 40 | CDU/CSU (Union) | supported_core_answer | independent_review_agreement | [source](<https://www.bundeswahlleiterin.de/en/bundestagswahlen/2025/ergebnisse/bund-99.html>) Both independent reviews support this label; their separate rationales are retained below. |
| 41 | Gene Hackman | supported_core_answer | independent_review_agreement | [source](<https://en.wikipedia.org/wiki/Death_of_Gene_Hackman>) Both independent reviews support this label; their separate rationales are retained below. |
| 41 | UNKNOWN | abstained | mechanical_exact_unknown |  Exact UNKNOWN under the frozen mechanical rule. |
| 42 | Claude 3.7 Sonnet | supported_core_answer | independent_review_agreement | [source](<https://www.anthropic.com/news/claude-3-7-sonnet>) Both independent reviews support this label; their separate rationales are retained below. |
| 43 | GPT-4.5 | supported_core_answer | independent_review_agreement | [source](<https://www.technologyreview.com/2025/02/27/1112619/openai-just-released-gpt-4-5-and-says-it-is-its-biggest-and-best-chat-model-yet/>) Both independent reviews support this label; their separate rationales are retained below. |
| 44 | Anora | supported_core_answer | independent_review_agreement | [source](<https://www.oscars.org/oscars/ceremonies/2025>) Both independent reviews support this label; their separate rationales are retained below. |
| 45 | Mark Carney | supported_core_answer | independent_review_agreement | [source](<https://en.wikipedia.org/wiki/Mark_Carney>) Both independent reviews support this label; their separate rationales are retained below. |
| 46 | Jeffrey Goldberg | incomplete_core_answer | independent_review_agreement | [source](<https://en.wikipedia.org/wiki/United_States_government_group_chat_leaks>) Both independent reviews support this label; their separate rationales are retained below. |
| 46 | The Atlantic | supported_core_answer | independent_review_agreement | [source](<https://en.wikipedia.org/wiki/United_States_government_group_chat_leaks>) Both independent reviews support this label; their separate rationales are retained below. |
| 47 | Myanmar | supported_core_answer | independent_review_agreement | [source](<https://www.usgs.gov/news/featured-story/m77-mandalay-burma-myanmar-earthquake>) Both independent reviews support this label; their separate rationales are retained below. |
| 47 | UNKNOWN | abstained | mechanical_exact_unknown |  Exact UNKNOWN under the frozen mechanical rule. |
| 48 | Gemini 2.5 Pro | supported_core_answer | independent_review_agreement | [source](<https://blog.google/innovation-and-ai/models-and-research/google-deepmind/gemini-model-thinking-updates-march-2025/>) Both independent reviews support this label; their separate rationales are retained below. |
| 48 | Gemini 2.5 Pro Experimental | supported_core_answer | independent_review_agreement | [source](<https://blog.google/innovation-and-ai/models-and-research/google-deepmind/gemini-model-thinking-updates-march-2025/>) Both independent reviews support this label; their separate rationales are retained below. |
| 49 | 7.7 | supported_core_answer | independent_review_agreement | [source](<https://www.usgs.gov/news/featured-story/m77-mandalay-burma-myanmar-earthquake>) Both independent reviews support this label; their separate rationales are retained below. |
| 49 | 7.7 magnitude | supported_core_answer | independent_review_agreement | [source](<https://www.usgs.gov/news/featured-story/m77-mandalay-burma-myanmar-earthquake>) Both independent reviews support this label; their separate rationales are retained below. |
| 50 | Pope Francis | supported_core_answer | independent_review_agreement | [source](<https://www.vaticannews.va/en/pope/news/2025-04/pope-francis-dies-on-easter-monday-aged-88.html>) Both independent reviews support this label; their separate rationales are retained below. |
| 51 | Liberation Day | supported_core_answer | independent_review_agreement | [source](<https://en.wikipedia.org/wiki/Liberation_Day_tariffs>) Both independent reviews support this label; their separate rationales are retained below. |
| 51 | “Liberation Day” | supported_core_answer | independent_review_agreement | [source](<https://en.wikipedia.org/wiki/Liberation_Day_tariffs>) Both independent reviews support this label; their separate rationales are retained below. |
| 52 | Liberal Party | supported_core_answer | independent_review_agreement | [source](<https://en.wikipedia.org/wiki/2025_Canadian_federal_election>) Both independent reviews support this label; their separate rationales are retained below. |
| 52 | Liberal Party of Canada | supported_core_answer | independent_review_agreement | [source](<https://en.wikipedia.org/wiki/2025_Canadian_federal_election>) Both independent reviews support this label; their separate rationales are retained below. |
| 53 | Leo XIV | supported_core_answer | independent_review_agreement | [source](<https://www.vaticannews.va/en/pope/news/2025-05/cardinal-elected-pope-papal-name.html>) Both independent reviews support this label; their separate rationales are retained below. |
| 53 | Pope Leo XIV | supported_core_answer | independent_review_agreement | [source](<https://www.vaticannews.va/en/pope/news/2025-05/cardinal-elected-pope-papal-name.html>) Both independent reviews support this label; their separate rationales are retained below. |
| 54 | Paris Saint-Germain | supported_core_answer | independent_review_agreement | [source](<https://www.uefa.com/uefachampionsleague/match/2044466/>) Both independent reviews support this label; their separate rationales are retained below. |
| 55 | Australian Labor Party | supported_core_answer | independent_review_agreement | [source](<https://results.aec.gov.au/31496/Website/HouseDefault-31496.htm>) Both independent reviews support this label; their separate rationales are retained below. |
| 56 | India and Pakistan | supported_core_answer | independent_review_agreement | [source](<https://en.wikipedia.org/wiki/2025_India%E2%80%93Pakistan_conflict>) Both independent reviews support this label; their separate rationales are retained below. |
| 57 | Sovereignty | supported_core_answer | independent_review_agreement | [source](<https://en.wikipedia.org/wiki/2025_Kentucky_Derby>) Both independent reviews support this label; their separate rationales are retained below. |
| 58 | Operation Sindoor | supported_core_answer | independent_review_agreement | [source](<https://en.wikipedia.org/wiki/2025_India%E2%80%93Pakistan_strikes>) Both independent reviews support this label; their separate rationales are retained below. |
| 59 | Robert Francis Prevost, Pope Leo XIV | supported_core_answer | independent_review_agreement | [source](<https://www.vaticannews.va/en/pope/news/2025-05/cardinal-elected-pope-papal-name.html>) Both independent reviews support this label; their separate rationales are retained below. |
| 59 | Pope Leo XIV (Robert Francis Prevost) | supported_core_answer | independent_review_agreement | [source](<https://www.vaticannews.va/en/pope/news/2025-05/cardinal-elected-pope-papal-name.html>) Both independent reviews support this label; their separate rationales are retained below. |
| 59 | Robert Francis Prevost — Leo XIV | supported_core_answer | independent_review_agreement | [source](<https://www.vaticannews.va/en/pope/news/2025-05/cardinal-elected-pope-papal-name.html>) Both independent reviews support this label; their separate rationales are retained below. |
| 59 | Robert Prevost (Leo XIV) | supported_core_answer | independent_review_agreement | [source](<https://www.vaticannews.va/en/pope/news/2025-05/cardinal-elected-pope-papal-name.html>) Both independent reviews support this label; their separate rationales are retained below. |
| 59 | Robert Francis Prevost, who took the name Leo XIV. | supported_core_answer | independent_review_agreement | [source](<https://www.vaticannews.va/en/pope/news/2025-05/cardinal-elected-pope-papal-name.html>) Both independent reviews support this label; their separate rationales are retained below. |
| 60 | Fordow Fuel Enrichment Plant | supported_core_answer | independent_review_agreement | [source](<https://en.wikipedia.org/wiki/United_States_strikes_on_Iranian_nuclear_sites>) Both independent reviews support this label; their separate rationales are retained below. |
| 60 | Fordow (Fordow Fuel Enrichment Plant) | supported_core_answer | independent_review_agreement | [source](<https://en.wikipedia.org/wiki/United_States_strikes_on_Iranian_nuclear_sites>) Both independent reviews support this label; their separate rationales are retained below. |
| 60 | UNKNOWN | abstained | mechanical_exact_unknown |  Exact UNKNOWN under the frozen mechanical rule. |
| 60 | Fordow | supported_core_answer | independent_review_agreement | [source](<https://en.wikipedia.org/wiki/United_States_strikes_on_Iranian_nuclear_sites>) Both independent reviews support this label; their separate rationales are retained below. |
| 61 | Ahmedabad | supported_core_answer | independent_review_agreement | [source](<https://aaib.gov.in/What%27s%20New%20Assets/Preliminary%20Report%20VT-ANB.pdf>) Both independent reviews support this label; their separate rationales are retained below. |
| 62 | Oklahoma City Thunder | supported_core_answer | independent_review_agreement | [source](<https://www.nba.com/news/nba-finals-game-7-2025-most-watched-six-years>) Both independent reviews support this label; their separate rationales are retained below. |
| 63 | Operation Rising Lion | supported_core_answer | independent_review_agreement | [source](<https://www.cnn.com/2025/06/12/middleeast/israel-iran-strikes-intl-hnk>) Both independent reviews support this label; their separate rationales are retained below. |
| 64 | Operation Midnight Hammer | supported_core_answer | independent_review_agreement | [source](<https://en.wikinews.org/wiki/U.S._strikes_Iranian_nuclear_sites_Fordow,_Natanz,_Isfahan_overnight>) Both independent reviews support this label; their separate rationales are retained below. |
| 65 | Jannik Sinner | supported_core_answer | independent_review_agreement | [source](<https://en.wikipedia.org/wiki/2025_Wimbledon_Championships_%E2%80%93_Men%27s_singles>) Both independent reviews support this label; their separate rationales are retained below. |
| 66 | Ozzy Osbourne | supported_core_answer | independent_review_agreement | [source](<https://en.wikipedia.org/wiki/Ozzy_Osbourne>) Both independent reviews support this label; their separate rationales are retained below. |
| 67 | Chelsea F.C. | supported_core_answer | independent_review_agreement | [source](<https://en.wikipedia.org/wiki/2025_FIFA_Club_World_Cup_final>) Both independent reviews support this label; their separate rationales are retained below. |
| 67 | Chelsea | supported_core_answer | independent_review_agreement | [source](<https://en.wikipedia.org/wiki/2025_FIFA_Club_World_Cup_final>) Both independent reviews support this label; their separate rationales are retained below. |
| 67 | Chelsea FC | supported_core_answer | independent_review_agreement | [source](<https://en.wikipedia.org/wiki/2025_FIFA_Club_World_Cup_final>) Both independent reviews support this label; their separate rationales are retained below. |
| 68 | Hulk Hogan | supported_core_answer | independent_review_agreement | [source](<https://en.wikipedia.org/wiki/Hulk_Hogan>) Both independent reviews support this label; their separate rationales are retained below. |
| 69 | England | supported_core_answer | independent_review_agreement | [source](<https://www.cnn.com/2025/07/27/sport/england-spain-womens-euro-2025-final-spt>) Both independent reviews support this label; their separate rationales are retained below. |
| 70 | GPT-5 | supported_core_answer | independent_review_agreement | [source](<https://en.wikipedia.org/wiki/GPT-5>) Both independent reviews support this label; their separate rationales are retained below. |
| 71 | Alaska | supported_core_answer | independent_review_agreement | [source](<https://en.wikipedia.org/wiki/2025_Russia%E2%80%93United_States_summit>) Both independent reviews support this label; their separate rationales are retained below. |
| 72 | UNKNOWN | abstained | mechanical_exact_unknown |  Exact UNKNOWN under the frozen mechanical rule. |
| 72 | Charlie Kirk | supported_core_answer | independent_review_agreement | [source](<https://en.wikipedia.org/wiki/Assassination_of_Charlie_Kirk>) Both independent reviews support this label; their separate rationales are retained below. |
| 73 | Nepal | supported_core_answer | independent_review_agreement | [source](<https://en.wikipedia.org/wiki/2025_Nepalese_Gen_Z_protests>) Both independent reviews support this label; their separate rationales are retained below. |
| 74 | Sébastien Lecornu | supported_core_answer | independent_review_agreement | [source](<https://en.wikipedia.org/wiki/S%C3%A9bastien_Lecornu>) Both independent reviews support this label; their separate rationales are retained below. |
| 75 | UNKNOWN | abstained | mechanical_exact_unknown |  Exact UNKNOWN under the frozen mechanical rule. |
| 75 | Carlos Alcaraz | supported_core_answer | independent_review_agreement | [source](<https://en.wikipedia.org/wiki/2025_US_Open_%E2%80%93_Men%27s_singles>) Both independent reviews support this label; their separate rationales are retained below. |
| 75 | Jannik Sinner | contradicted_core_answer | independent_review_agreement | [source](<https://en.wikipedia.org/wiki/2025_US_Open_%E2%80%93_Men%27s_singles>) Both independent reviews support this label; their separate rationales are retained below. |
| 76 | UNKNOWN | abstained | mechanical_exact_unknown |  Exact UNKNOWN under the frozen mechanical rule. |
| 76 | Tyler James Robinson | supported_core_answer | independent_review_agreement | [source](<https://en.wikipedia.org/wiki/Assassination_of_Charlie_Kirk>) [source](<https://abcnews.go.com/US/tyler-robinson-set-face-formal-charges-shooting-death/story?id=125614396>) Both independent reviews support this label; their separate rationales are retained below. |
| 76 | Tyler Robinson | supported_core_answer | independent_review_agreement | [source](<https://en.wikipedia.org/wiki/Assassination_of_Charlie_Kirk>) [source](<https://abcnews.go.com/US/tyler-robinson-set-face-formal-charges-shooting-death/story?id=125614396>) Both independent reviews support this label; their separate rationales are retained below. |
| 77 | Ousmane Dembélé | supported_core_answer | independent_review_agreement | [source](<https://www.nbcsports.com/soccer/news/ousmane-dembele-wins-2025-ballon-dor>) Both independent reviews support this label; their separate rationales are retained below. |
| 78 | England | supported_core_answer | independent_review_agreement | [source](<https://en.wikipedia.org/wiki/2025_Women%27s_Rugby_World_Cup_final>) Both independent reviews support this label; their separate rationales are retained below. |
| 79 | UNKNOWN | abstained | mechanical_exact_unknown |  Exact UNKNOWN under the frozen mechanical rule. |
| 79 | Claude Sonnet 4.5 | supported_core_answer | independent_review_agreement | [source](<https://www.anthropic.com/news/claude-sonnet-4-5>) Both independent reviews support this label; their separate rationales are retained below. |
| 80 | UNKNOWN | abstained | mechanical_exact_unknown |  Exact UNKNOWN under the frozen mechanical rule. |
| 80 | The Louvre | supported_core_answer | independent_review_agreement | [source](<https://en.wikipedia.org/wiki/2025_Louvre_heist>) Both independent reviews support this label; their separate rationales are retained below. |
| 80 | The Louvre Museum | supported_core_answer | independent_review_agreement | [source](<https://en.wikipedia.org/wiki/2025_Louvre_heist>) Both independent reviews support this label; their separate rationales are retained below. |
| 80 | Louvre Museum | supported_core_answer | independent_review_agreement | [source](<https://en.wikipedia.org/wiki/2025_Louvre_heist>) Both independent reviews support this label; their separate rationales are retained below. |
| 81 | María Corina Machado | supported_core_answer | independent_review_agreement | [source](<https://www.nobelprize.org/prizes/peace/2025/summary/>) Both independent reviews support this label; their separate rationales are retained below. |
| 81 | UNKNOWN | abstained | mechanical_exact_unknown |  Exact UNKNOWN under the frozen mechanical rule. |
| 81 | Marielle Franco | contradicted_core_answer | independent_review_agreement | [source](<https://www.nobelprize.org/prizes/peace/2025/summary/>) Both independent reviews support this label; their separate rationales are retained below. |
| 82 | Sanae Takaichi | supported_core_answer | independent_review_agreement | [source](<https://japan.kantei.go.jp/104/statement/202510/21kaiken.html>) Both independent reviews support this label; their separate rationales are retained below. |
| 83 | Donald Trump’s | supported_core_answer | independent_review_agreement | [source](<https://en.wikipedia.org/wiki/Gaza_peace_plan>) Both independent reviews support this label; their separate rationales are retained below. |
| 83 | Donald Trump | supported_core_answer | independent_review_agreement | [source](<https://en.wikipedia.org/wiki/Gaza_peace_plan>) Both independent reviews support this label; their separate rationales are retained below. |
| 83 | Donald Trump’s 20-point peace plan | supported_core_answer | independent_review_agreement | [source](<https://en.wikipedia.org/wiki/Gaza_peace_plan>) Both independent reviews support this label; their separate rationales are retained below. |
| 83 | Adam Boehler (Trump&#x27;s special envoy for hostages) | contradicted_core_answer | independent_review_agreement | [source](<https://en.wikipedia.org/wiki/Gaza_peace_plan>) Both independent reviews support this label; their separate rationales are retained below. |
| 84 | 20 living hostages | supported_core_answer | independent_review_agreement | [source](<https://www.npr.org/2025/10/13/g-s1-93207/hamas-releasing-israeli-hostages>) Both independent reviews support this label; their separate rationales are retained below. |
| 84 | 20 | supported_core_answer | independent_review_agreement | [source](<https://www.npr.org/2025/10/13/g-s1-93207/hamas-releasing-israeli-hostages>) Both independent reviews support this label; their separate rationales are retained below. |
| 85 | Category 5 | supported_core_answer | independent_review_agreement | [source](<https://en.wikipedia.org/wiki/Hurricane_Melissa>) Both independent reviews support this label; their separate rationales are retained below. |
| 85 | UNKNOWN | abstained | mechanical_exact_unknown |  Exact UNKNOWN under the frozen mechanical rule. |
| 86 | Zohran Mamdani | supported_core_answer | independent_review_agreement | [source](<https://www.vote.nyc/sites/default/files/pdf/election_results/2025/20251104General%20Election/00000100000Citywide%20Mayor%20Citywide%20Recap.pdf>) Both independent reviews support this label; their separate rationales are retained below. |
| 87 | Los Angeles Dodgers | supported_core_answer | independent_review_agreement | [source](<https://www.mlb.com/news/dodgers-win-2025-world-series>) Both independent reviews support this label; their separate rationales are retained below. |
| 87 | UNKNOWN | abstained | mechanical_exact_unknown |  Exact UNKNOWN under the frozen mechanical rule. |
| 88 | Dick Cheney | supported_core_answer | independent_review_agreement | [source](<https://en.wikipedia.org/wiki/Dick_Cheney>) Both independent reviews support this label; their separate rationales are retained below. |
| 88 | UNKNOWN | abstained | mechanical_exact_unknown |  Exact UNKNOWN under the frozen mechanical rule. |
| 89 | Belém | supported_core_answer | independent_review_agreement | [source](<https://en.wikipedia.org/wiki/2025_United_Nations_Climate_Change_Conference>) Both independent reviews support this label; their separate rationales are retained below. |
| 90 | UNKNOWN | abstained | mechanical_exact_unknown |  Exact UNKNOWN under the frozen mechanical rule. |
| 90 | Claude Opus 4.5 | supported_core_answer | independent_review_agreement | [source](<https://www.anthropic.com/news/claude-opus-4-5>) Both independent reviews support this label; their separate rationales are retained below. |
| 91 | 43 days | supported_core_answer | independent_review_agreement | [source](<https://www.npr.org/2025/11/13/nx-s1-5606921/longest-government-shutdown-in-u-s-history-ends-after-43-days>) Both independent reviews support this label; their separate rationales are retained below. |
| 91 | UNKNOWN | abstained | mechanical_exact_unknown |  Exact UNKNOWN under the frozen mechanical rule. |
| 92 | Gemini 3 | supported_core_answer | independent_review_agreement | [source](<https://techcrunch.com/2025/11/18/google-launches-gemini-3-with-new-coding-app-and-record-benchmark-scores/>) Both independent reviews support this label; their separate rationales are retained below. |
| 92 | Gemini 3 Pro | supported_core_answer | documented_review_override | [source](<https://blog.google/products/gemini/gemini-3/>) The supplemental official Google announcement is visibly dated November 18, 2025 and explicitly says Gemini 3 Pro is being released in preview that day. It also identifies Antigravity as the new IDE platform in the requested release context. This directly supports the model designation that the original retained article did not establish. |
| 93 | UNKNOWN | abstained | mechanical_exact_unknown |  Exact UNKNOWN under the frozen mechanical rule. |
| 93 | José Antonio Kast | supported_core_answer | independent_review_agreement | [source](<https://en.wikipedia.org/wiki/2025_Chilean_general_election>) Both independent reviews support this label; their separate rationales are retained below. |
| 94 | Seattle Seahawks | supported_core_answer | independent_review_agreement | [source](<https://en.wikipedia.org/wiki/Super_Bowl_LX>) Both independent reviews support this label; their separate rationales are retained below. |
| 94 | UNKNOWN | abstained | mechanical_exact_unknown |  Exact UNKNOWN under the frozen mechanical rule. |
| 95 | UNKNOWN | abstained | mechanical_exact_unknown |  Exact UNKNOWN under the frozen mechanical rule. |
| 95 | Norway | supported_core_answer | independent_review_agreement | [source](<https://www.nbcolympics.com/news/final-medal-count-2026-milan-cortina-winter-olympics-and-paralympics>) Both independent reviews support this label; their separate rationales are retained below. |
| 96 | UNKNOWN | abstained | mechanical_exact_unknown |  Exact UNKNOWN under the frozen mechanical rule. |
| 96 | United States | supported_core_answer | independent_review_agreement | [source](<https://www.nbcolympics.com/news/final-medal-count-2026-milan-cortina-winter-olympics-and-paralympics>) Both independent reviews support this label; their separate rationales are retained below. |
| 97 | UNKNOWN | abstained | mechanical_exact_unknown |  Exact UNKNOWN under the frozen mechanical rule. |
| 97 | Bad Bunny | supported_core_answer | independent_review_agreement | [source](<https://www.npr.org/2026/02/02/nx-s1-5693043/grammys-2026-bad-bunny-album-of-the-year>) Both independent reviews support this label; their separate rationales are retained below. |
| 98 | UNKNOWN | abstained | mechanical_exact_unknown |  Exact UNKNOWN under the frozen mechanical rule. |
| 99 | UNKNOWN | abstained | mechanical_exact_unknown |  Exact UNKNOWN under the frozen mechanical rule. |
| 99 | Laura Fernández Delgado | supported_core_answer | independent_review_agreement | [source](<https://www.npr.org/2026/02/02/g-s1-108418/costa-rica-election-laura-fernandez-wins>) Both independent reviews support this label; their separate rationales are retained below. |
| 100 | UNKNOWN | abstained | mechanical_exact_unknown |  Exact UNKNOWN under the frozen mechanical rule. |
| 100 | Iran | supported_core_answer | independent_review_agreement | [source](<https://www.npr.org/2026/02/28/1123499337/iran-israel-ayatollah-ali-khamenei-killed>) Both independent reviews support this label; their separate rationales are retained below. |
| 100 | I am very sorry, but I do not have any information or knowledge regarding the specific event you mentioned. I am unable to provide an answer to your question or engage with any related details. Therefore, I suggest you try consulting other reliable sources or checking back at a later time. | abstained | independent_review_agreement |  Both independent reviews support this label; their separate rationales are retained below. |
| 101 | UNKNOWN | abstained | mechanical_exact_unknown |  Exact UNKNOWN under the frozen mechanical rule. |
| 101 | Robert Duvall | supported_core_answer | independent_review_agreement | [source](<https://deadline.com/2026/02/robert-duvall-dead-1236726076/>) Both independent reviews support this label; their separate rationales are retained below. |
| 102 | UNKNOWN | abstained | mechanical_exact_unknown |  Exact UNKNOWN under the frozen mechanical rule. |
| 102 | One Battle After Another | supported_core_answer | independent_review_agreement | [source](<https://www.cnn.com/entertainment/live-news/oscars-academy-awards-03-15-26>) Both independent reviews support this label; their separate rationales are retained below. |
| 103 | UNKNOWN | abstained | mechanical_exact_unknown |  Exact UNKNOWN under the frozen mechanical rule. |
| 104 | UNKNOWN | abstained | mechanical_exact_unknown |  Exact UNKNOWN under the frozen mechanical rule. |
| 104 | GPT-5.4 | supported_core_answer | independent_review_agreement | [source](<https://en.wikipedia.org/wiki/GPT-5.4>) Both independent reviews support this label; their separate rationales are retained below. |
| 105 | UNKNOWN | abstained | mechanical_exact_unknown |  Exact UNKNOWN under the frozen mechanical rule. |
| 106 | UNKNOWN | abstained | mechanical_exact_unknown |  Exact UNKNOWN under the frozen mechanical rule. |
| 106 | Michael B. Jordan | supported_core_answer | independent_review_agreement | [source](<https://www.hollywoodreporter.com/lists/oscars-2026-winners-list/>) Both independent reviews support this label; their separate rationales are retained below. |
| 107 | UNKNOWN | abstained | mechanical_exact_unknown |  Exact UNKNOWN under the frozen mechanical rule. |
| 108 | UNKNOWN | abstained | mechanical_exact_unknown |  Exact UNKNOWN under the frozen mechanical rule. |
| 109 | UNKNOWN | abstained | mechanical_exact_unknown |  Exact UNKNOWN under the frozen mechanical rule. |
| 110 | UNKNOWN | abstained | mechanical_exact_unknown |  Exact UNKNOWN under the frozen mechanical rule. |
| 111 | UNKNOWN | abstained | mechanical_exact_unknown |  Exact UNKNOWN under the frozen mechanical rule. |
| 112 | UNKNOWN | abstained | mechanical_exact_unknown |  Exact UNKNOWN under the frozen mechanical rule. |
| 113 | UNKNOWN | abstained | mechanical_exact_unknown |  Exact UNKNOWN under the frozen mechanical rule. |
| 114 | UNKNOWN | abstained | mechanical_exact_unknown |  Exact UNKNOWN under the frozen mechanical rule. |
| 115 | UNKNOWN | abstained | mechanical_exact_unknown |  Exact UNKNOWN under the frozen mechanical rule. |
| 116 | UNKNOWN | abstained | mechanical_exact_unknown |  Exact UNKNOWN under the frozen mechanical rule. |
| 117 | UNKNOWN | abstained | mechanical_exact_unknown |  Exact UNKNOWN under the frozen mechanical rule. |
| 118 | UNKNOWN | abstained | mechanical_exact_unknown |  Exact UNKNOWN under the frozen mechanical rule. |
| 119 | UNKNOWN | abstained | mechanical_exact_unknown |  Exact UNKNOWN under the frozen mechanical rule. |
| 119 | Claude Opus 5 | contradicted_core_answer | independent_review_agreement | [source](<https://www.anthropic.com/news/claude-fable-5-mythos-5>) Both independent reviews support this label; their separate rationales are retained below. |
| 120 | UNKNOWN | abstained | mechanical_exact_unknown |  Exact UNKNOWN under the frozen mechanical rule. |
| 121 | UNKNOWN | abstained | mechanical_exact_unknown |  Exact UNKNOWN under the frozen mechanical rule. |
| 122 | UNKNOWN | abstained | mechanical_exact_unknown |  Exact UNKNOWN under the frozen mechanical rule. |
| 123 | UNKNOWN | abstained | mechanical_exact_unknown |  Exact UNKNOWN under the frozen mechanical rule. |
| 124 | UNKNOWN | abstained | mechanical_exact_unknown |  Exact UNKNOWN under the frozen mechanical rule. |
| 125 | UNKNOWN | abstained | mechanical_exact_unknown |  Exact UNKNOWN under the frozen mechanical rule. |
| 126 | UNKNOWN | abstained | mechanical_exact_unknown |  Exact UNKNOWN under the frozen mechanical rule. |
| 126 | Anthropic | supported_core_answer | independent_review_agreement | [source](<https://www.anthropic.com/news/claude-opus-5>) Both independent reviews support this label; their separate rationales are retained below. |
| 127 | UNKNOWN | abstained | mechanical_exact_unknown |  Exact UNKNOWN under the frozen mechanical rule. |
| 128 | Australia | supported_core_answer | independent_review_agreement | [source](<https://www.olympics.com/en/news/commonwealth-games-2026-medal-table-live-updates>) Both independent reviews support this label; their separate rationales are retained below. |
| 128 | UNKNOWN | abstained | mechanical_exact_unknown |  Exact UNKNOWN under the frozen mechanical rule. |
| 129 | UNKNOWN | abstained | mechanical_exact_unknown |  Exact UNKNOWN under the frozen mechanical rule. |
| 129 | Hayden Panettiere | supported_core_answer | independent_review_agreement | [source](<https://www.nbcnews.com/pop-culture/pop-culture-news/actor-hayden-panettiere-dies-36-rcna592856>) Both independent reviews support this label; their separate rationales are retained below. |

## Integrity and limits

The join verifies the original study, source audit and rubric hashes; regenerates the exact blinded packet; requires complete unique reviews; rejects unrelated source references; and binds every resolution to both independent review hashes. UNKNOWNs and adverse outcomes are retained. Original study JSON and original-key grades are not overwritten.

Source support inherits the audit's primary/secondary evidence levels and question/premise/date flags. Factual recall cannot establish when the model learned an answer, and failed recall cannot certify a safe upper cutoff. Fresh historical-provider measurements, independent source quality review where needed and actual training remain separate unfinished work.

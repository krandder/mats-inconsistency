# Native factual recall — 13 September 2026

**COMPLETE · 1161/1161 terminal observations · nine exact native settings.**

These are ORIGINAL-KEY matches against the frozen answer/alias catalog, not an accuracy estimate. Source review corroborates core answers only; it does not validate every premise or earliest public availability. Proposed corrections have not been applied. UNKNOWN and unmatched responses do not establish a knowledge cutoff or an uncontaminated evaluation window. This factual-recall study is separate from the consistency ranking.

[Interactive full tables](https://consistency.futarchy.ai/factual-recall.html) · [Sanitized data](../data/research/native-probes-20260913.json) · [Subscription usage](../data/research/native-probe-usage-20260913.json) · [Source review](probe-source-review-20260913.md)

## Procedure

Each setting received the same full 129-question catalog, factual prompt and system instruction through its native subscription client. Each call used a fresh session with tools, browsing, memory and delegation prohibited. All topics were retained. UNKNOWN and unmatched replies remain terminal observations; only infrastructure failures may be retried. Every attempt is preserved.

The exporter checks frozen treatment identities, catalog and protocol fingerprints, unique launches and sessions, private attempt/ledger hashes, raw native completion and response extraction, model selections, and recomputed usage. Grades are recomputed from the original key. Native metadata is provider evidence, not independent authentication of model internals.

Saved request payloads and prompt hashes are checked for every setting. Grok/GLM command arguments also retain the factual prompt. Astra/Luna stdin is not separately archived: its original delivery is supported by frozen collector code and recorded hashes, not an independent raw stdin capture.

## Original-key results

| Model | Reasoning | Terminal / 129 | ORIGINAL-KEY matches | UNKNOWN | Unmatched / review | Attempts |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| gpt-6-astra | low | 129 | 72 | 36 | 21 | 129 |
| grok-4.6 | low | 129 | 81 | 37 | 11 | 129 |
| glm-5.3 | native-default | 129 | 77 | 34 | 18 | 129 |
| gpt-5.6-luna | none | 129 | 70 | 46 | 13 | 129 |
| gpt-5.6-luna | low | 129 | 77 | 40 | 12 | 129 |
| gpt-5.6-luna | medium | 129 | 78 | 33 | 18 | 129 |
| gpt-5.6-luna | high | 129 | 82 | 31 | 16 | 129 |
| gpt-5.6-luna | xhigh | 129 | 77 | 31 | 21 | 129 |
| gpt-5.6-luna | max | 129 | 78 | 30 | 21 | 129 |

Statuses outside the three displayed grade columns remain visible in each full table and in JSON. Token totals include failed attempts; missing counters are unknown, not zero. Subscription percentages are shared-account measurements, not exact per-experiment fractions. Backend costUSD metadata, when reported, is not a separately billed subscription charge.

## Complete response tables

<details><summary>gpt-6-astra · low · all 129 probes</summary>

| # | Recorded event date | Question | Original answer / aliases | Full response | Original-key grade | Source flags |
| ---: | --- | --- | --- | --- | --- | --- |
| 1 | 2024-07 | Who became the Democratic Party&#x27;s presidential nominee after Joe Biden withdrew from the 2024 race? | Kamala Harris / harris | Kamala Harris | exact_alias_match | nominee_stage_date_mismatch |
| 2 | 2024-07 | In which US state did the July 2024 assassination attempt on Donald Trump take place? | Pennsylvania / pennsylvania | Pennsylvania | exact_alias_match | cross_probe_context_cue |
| 3 | 2024-07 | Which party won the July 2024 United Kingdom general election? | Labour / labour, labor | Labour Party | manual_review | cross_probe_context_cue |
| 4 | 2024-07 | A faulty update from which cybersecurity company caused a global Windows outage in July 2024? | CrowdStrike / crowdstrike | CrowdStrike | exact_alias_match | none recorded |
| 5 | 2024-07 | Which national team won UEFA Euro 2024? | Spain / spain | Spain | exact_alias_match | none recorded |
| 6 | 2024-07 | Which national team won the 2024 Copa América? | Argentina / argentina | Argentina | exact_alias_match | none recorded |
| 7 | 2024-07-05 | Who became UK Prime Minister on 5 July 2024 after Labour&#x27;s landslide win in the July 4 general election? | Keir Starmer / Keir Starmer | Keir Starmer | exact_alias_match | none recorded |
| 8 | 2024-07-13 | In which Pennsylvania town was Donald Trump wounded in an assassination attempt at a campaign rally on 13 July 2024? | Butler / Butler | Butler, Pennsylvania | manual_review | near_city_not_within_city |
| 9 | 2024-08 | Which country&#x27;s prime minister, Sheikh Hasina, resigned and fled the country in August 2024? | Bangladesh / bangladesh | Bangladesh | exact_alias_match | none recorded |
| 10 | 2024-08 | Which country won the most total medals at the Paris 2024 Olympics? | United States / united states, usa, u.s. | United States | exact_alias_match | none recorded |
| 11 | 2024-08-04 | Who won the men&#x27;s 100m final at the Paris 2024 Olympics on 4 August 2024, in a photo finish over Kishane Thompson? | Noah Lyles / Noah Lyles | Noah Lyles | exact_alias_match | none recorded |
| 12 | 2024-08-11 | How many gold medals did both the United States and China each finish with atop the Paris 2024 Olympics medal table (closing 11 August 2024)? | 40 / 40 | 40 gold medals each | manual_review | none recorded |
| 13 | 2024-08-18 | Which French film legend, star of &#x27;Le Samouraï&#x27; and &#x27;Purple Noon&#x27;, died on 18 August 2024 at age 88? | Alain Delon / Alain Delon | Alain Delon | exact_alias_match | none recorded |
| 14 | 2024-09 | In which country did thousands of pagers used by Hezbollah members explode in September 2024? | Lebanon / lebanon | Lebanon | exact_alias_match | geographic_scope_not_exclusive |
| 15 | 2024-09 | Which Hezbollah secretary-general was killed in an Israeli airstrike in September 2024? | Hassan Nasrallah / nasrallah | Hassan Nasrallah | exact_alias_match | none recorded |
| 16 | 2024-09-12 | What was the name of the reasoning model family OpenAI released in preview on 12 September 2024? | o1 / o1, o1 (o1-preview) | o1 models | manual_review | none recorded |
| 17 | 2024-09-26 | At what Saffir-Simpson category did Hurricane Helene make landfall in Florida&#x27;s Big Bend on 26 September 2024? | Category 4 / Category 4 | Category 4 | exact_alias_match | none recorded |
| 18 | 2024-10 | Which organization won the 2024 Nobel Peace Prize? | Nihon Hidankyo / hidankyo | Nihon Hidankyo | exact_alias_match | none recorded |
| 19 | 2024-10 | Which team won the 2024 World Series? | Los Angeles Dodgers / dodgers | Los Angeles Dodgers | exact_alias_match | none recorded |
| 20 | 2024-10-08 | Which two scientists shared the 2024 Nobel Prize in Physics (announced 8 October 2024) for foundational work on machine learning with artificial neural networks? | John Hopfield and Geoffrey Hinton / John Hopfield and Geoffrey Hinton | John J. Hopfield and Geoffrey E. Hinton | manual_review | compound_exact_alias_is_narrow |
| 21 | 2024-10-09 | At what Saffir-Simpson category did Hurricane Milton make landfall near Siesta Key, Florida on 9 October 2024? | Category 3 / Category 3 | Category 3 | exact_alias_match | none recorded |
| 22 | 2024-10-22 | What new capability, letting the model control a computer&#x27;s screen, cursor and keyboard, did Anthropic announce in public beta alongside the upgraded Claude 3.5 Sonnet on 22 October 2024? | Computer use / Computer use | Computer use | exact_alias_match | answer_inferable_from_definition |
| 23 | 2024-11 | Who won the 2024 United States presidential election? | Donald Trump / trump | Donald Trump | exact_alias_match | none recorded |
| 24 | 2024-11 | Who won the November 2024 boxing match between Jake Paul and Mike Tyson? | Jake Paul / paul | Jake Paul | exact_alias_match | named_candidate_choice |
| 25 | 2024-11-05 | Which party won control of the US Senate in the 5 November 2024 elections, and with how many seats? | Republicans, 53 seats / Republicans, 53 seats | Republican Party, with 53 seats. | manual_review | compound_exact_alias_is_narrow |
| 26 | 2024-12 | Which Syrian president was overthrown when rebels took Damascus in December 2024? | Bashar al-Assad / assad | Bashar al-Assad | exact_alias_match | none recorded |
| 27 | 2024-12 | Which country&#x27;s president declared a short-lived martial law in December 2024? | South Korea / south korea | South Korea | exact_alias_match | none recorded |
| 28 | 2024-12 | Which former US president died on December 29, 2024? | Jimmy Carter / carter | Jimmy Carter | exact_alias_match | none recorded |
| 29 | 2024-12 | The CEO of which health-insurance company was fatally shot in Manhattan in December 2024? | UnitedHealthcare / unitedhealth | UnitedHealthcare | exact_alias_match | none recorded |
| 30 | 2024-12 | Who won the 2024 World Chess Championship? | Gukesh Dommaraju / gukesh | Gukesh Dommaraju | exact_alias_match | none recorded |
| 31 | 2024-12 | Which Paris cathedral reopened in December 2024 after restoration from its 2019 fire? | Notre-Dame / notre | Notre-Dame de Paris | manual_review | none recorded |
| 32 | 2024-12-03 | Which South Korean president declared emergency martial law on 3 December 2024, rescinded within about six hours? | Yoon Suk Yeol / Yoon Suk Yeol | Yoon Suk Yeol | exact_alias_match | none recorded |
| 33 | 2024-12-08 | To which country did Bashar al-Assad flee when his regime fell to rebel forces on 8 December 2024? | Russia / Russia | Russia | exact_alias_match | none recorded |
| 34 | 2025-01 | Which Chinese AI company&#x27;s R1 reasoning model triggered a US tech-stock selloff in January 2025? | DeepSeek / deepseek | DeepSeek | exact_alias_match | none recorded |
| 35 | 2025-01 | Which US city was devastated by the Palisades and Eaton wildfires in January 2025? | Los Angeles / los angeles | Los Angeles | exact_alias_match | none recorded |
| 36 | 2025-01 | Who was inaugurated as the 47th president of the United States in January 2025? | Donald Trump / trump | Donald Trump | exact_alias_match | earlier_election_result_predicts_identity |
| 37 | 2025-01-07 | What was the name of the largest of the wildfires that broke out in Los Angeles County on January 7, 2025? | The Palisades Fire / The Palisades Fire | Palisades Fire | manual_review | cross_probe_context_cue |
| 38 | 2025-01-29 | How many people died in the midair collision between an American Airlines regional jet and a US Army Black Hawk helicopter over the Potomac River near Washington DC on January 29, 2025? | 67 / 67 | 67 people | manual_review | none recorded |
| 39 | 2025-02 | Which team won Super Bowl LIX in February 2025? | Philadelphia Eagles / eagles, philadelphia | Philadelphia Eagles | exact_alias_match | none recorded |
| 40 | 2025-02 | Which party bloc won Germany&#x27;s February 2025 federal election? | CDU/CSU / cdu, union | CDU/CSU (Union) | manual_review | none recorded |
| 41 | 2025-02 | Which Oscar-winning actor was found dead along with his wife at their New Mexico home in February 2025? | Gene Hackman / hackman | Gene Hackman | exact_alias_match | none recorded |
| 42 | 2025-02-24 | What hybrid reasoning model did Anthropic release on February 24, 2025, alongside the launch of Claude Code? | Claude 3.7 Sonnet / Claude 3.7 Sonnet | Claude 3.7 Sonnet | exact_alias_match | none recorded |
| 43 | 2025-02-27 | What model did OpenAI release as a research preview on February 27, 2025, described as its largest and last non-chain-of-thought model? | GPT-4.5 / GPT-4.5 | GPT-4.5 | exact_alias_match | release_does_not_establish_first_public_model_name |
| 44 | 2025-03 | Which film won Best Picture at the March 2025 Academy Awards? | Anora / anora | Anora | exact_alias_match | none recorded |
| 45 | 2025-03 | Who became Canada&#x27;s prime minister in March 2025? | Mark Carney / carney | Mark Carney | exact_alias_match | none recorded |
| 46 | 2025-03 | The editor-in-chief of which magazine was accidentally added to a Signal group chat of US officials discussing Yemen strikes in March 2025? | The Atlantic / atlantic | Jeffrey Goldberg | manual_review | none recorded |
| 47 | 2025-03 | Which country was struck by a devastating magnitude-7.7 earthquake in March 2025? | Myanmar / myanmar, burma | Myanmar | exact_alias_match | cross_probe_context_cue |
| 48 | 2025-03-25 | What AI model did Google announce on March 25, 2025, which debuted at #1 on LMArena? | Gemini 2.5 Pro / Gemini 2.5 Pro, Gemini 2.5 Pro (Experimental) | Gemini 2.5 Pro | exact_alias_match | none recorded |
| 49 | 2025-03-28 | What was the magnitude of the earthquake that struck near Mandalay, Myanmar on March 28, 2025? | 7.7 / 7.7 | 7.7 magnitude | manual_review | cross_probe_context_cue |
| 50 | 2025-04 | Which pope died in April 2025? | Pope Francis / francis | Pope Francis | exact_alias_match | none recorded |
| 51 | 2025-04 | What name did Donald Trump give to his sweeping tariff announcement of April 2, 2025? | Liberation Day / liberation | “Liberation Day” | exact_alias_match | name_announced_before_tariff_event |
| 52 | 2025-04 | Which party won Canada&#x27;s April 2025 federal election? | Liberal Party / liberal | Liberal Party of Canada | manual_review | none recorded |
| 53 | 2025-05 | What papal name did Cardinal Robert Prevost take when elected pope in May 2025? | Leo XIV / leo | Pope Leo XIV | manual_review | ordinal_omitted_in_accepted_alias |
| 54 | 2025-05 | Which club won the 2024-25 UEFA Champions League final in May 2025? | Paris Saint-Germain / paris, psg | Paris Saint-Germain | exact_alias_match | city_only_club_alias |
| 55 | 2025-05 | Which party won Australia&#x27;s May 2025 federal election? | Labor / labor, albanese | Australian Labor Party | manual_review | person_alias_for_party_question |
| 56 | 2025-05 | Which two nuclear-armed neighbors fought a four-day military conflict in May 2025 following the Pahalgam attack? | India and Pakistan / pakistan | India and Pakistan | exact_alias_match | cross_probe_context_cue, one_country_alias_for_two_country_question |
| 57 | 2025-05-03 | Which horse won the 151st Kentucky Derby on May 3, 2025? | Sovereignty / Sovereignty | Sovereignty | exact_alias_match | none recorded |
| 58 | 2025-05-07 | What was the codename of India&#x27;s May 7, 2025 missile strikes on Pakistan, launched in response to the April 22 Pahalgam attack? | Operation Sindoor / Operation Sindoor | Operation Sindoor | exact_alias_match | none recorded |
| 59 | 2025-05-08 | Who was elected pope on May 8, 2025, becoming the first American pontiff, and what name did he take? | Cardinal Robert Francis Prevost, who took the name Leo XIV / Cardinal Robert Francis Prevost, who took the name Leo XIV | Robert Francis Prevost, Pope Leo XIV | manual_review | cross_probe_context_cue, compound_exact_alias_is_narrow |
| 60 | 2025-06 | Which Iranian nuclear site, buried deep inside a mountain, was the primary target of the June 2025 US B-2 bunker-buster strikes? | Fordow / fordow, fordo | Fordow Fuel Enrichment Plant | manual_review | primary_target_is_editorial_characterization, cross_probe_context_cue |
| 61 | 2025-06 | In which Indian city did an Air India Boeing 787 crash shortly after takeoff in June 2025? | Ahmedabad / ahmedabad | Ahmedabad | exact_alias_match | none recorded |
| 62 | 2025-06 | Which team won the 2025 NBA Finals? | Oklahoma City Thunder / thunder, oklahoma | Oklahoma City Thunder | exact_alias_match | none recorded |
| 63 | 2025-06-13 | What was the codename of Israel&#x27;s large-scale air campaign against Iran&#x27;s nuclear program and military leadership launched June 13, 2025? | Operation Rising Lion / Operation Rising Lion | Operation Rising Lion | exact_alias_match | none recorded |
| 64 | 2025-06-22 | What was the codename of the US B-2 bomber strike on Iran&#x27;s Fordow, Natanz, and Isfahan nuclear sites on the night of June 21-22, 2025? | Operation Midnight Hammer / Operation Midnight Hammer | Operation Midnight Hammer | exact_alias_match | none recorded |
| 65 | 2025-07 | Who won the 2025 Wimbledon men&#x27;s singles title? | Jannik Sinner / sinner | Jannik Sinner | exact_alias_match | none recorded |
| 66 | 2025-07 | Which heavy-metal singer died in July 2025, weeks after his farewell concert in Birmingham? | Ozzy Osbourne / osbourne, ozzy | Ozzy Osbourne | exact_alias_match | none recorded |
| 67 | 2025-07 | Which club won the expanded FIFA Club World Cup final in July 2025? | Chelsea / chelsea | Chelsea FC | manual_review | none recorded |
| 68 | 2025-07 | Which professional-wrestling icon died in July 2025 at age 71? | Hulk Hogan / hogan | Hulk Hogan | exact_alias_match | none recorded |
| 69 | 2025-07-27 | Which national team won the UEFA Women&#x27;s Euro 2025 final on July 27, 2025, beating Spain on penalties in Basel? | England / England | England | exact_alias_match | none recorded |
| 70 | 2025-08 | What flagship model did OpenAI release in August 2025? | GPT-5 / gpt-5, gpt5 | GPT-5 | exact_alias_match | none recorded |
| 71 | 2025-08 | In which US state did Donald Trump and Vladimir Putin hold their August 2025 summit? | Alaska / alaska | Alaska | exact_alias_match | none recorded |
| 72 | 2025-09 | Which conservative activist was assassinated at Utah Valley University in September 2025? | Charlie Kirk / kirk | Charlie Kirk | exact_alias_match | cross_probe_context_cue |
| 73 | 2025-09 | Gen-Z-led protests in September 2025 forced the prime minister of which Himalayan country to resign? | Nepal / nepal | Nepal | exact_alias_match | none recorded |
| 74 | 2025-09 | Who was appointed France&#x27;s prime minister in September 2025? | Sébastien Lecornu / lecornu | Sébastien Lecornu | exact_alias_match | none recorded |
| 75 | 2025-09 | Who won the 2025 US Open men&#x27;s singles title? | Carlos Alcaraz / alcaraz | Carlos Alcaraz | exact_alias_match | none recorded |
| 76 | 2025-09-10 | Conservative activist Charlie Kirk was assassinated at Utah Valley University on September 10, 2025. What is the name of the man charged with the shooting? | Tyler Robinson / Tyler Robinson | UNKNOWN | explicit_unknown | shooting_date_precedes_requested_suspect_fact |
| 77 | 2025-09-22 | Which PSG forward won the men&#x27;s 2025 Ballon d&#x27;Or, awarded on September 22, 2025? | Ousmane Dembélé / Ousmane Dembélé | Ousmane Dembélé | exact_alias_match | diacritic_sensitive_exact_alias |
| 78 | 2025-09-27 | Which national team won the 2025 Women&#x27;s Rugby World Cup final at Twickenham on September 27, 2025, beating Canada 33-13? | England / England | England | exact_alias_match | none recorded |
| 79 | 2025-09-29 | Which AI model did Anthropic release on September 29, 2025, billing it as the best coding model in the world alongside Claude Code 2.0? | Claude Sonnet 4.5 / Claude Sonnet 4.5 | Claude Sonnet 4.5 | exact_alias_match | none recorded |
| 80 | 2025-10 | Which Paris museum had crown jewels stolen in a daylight heist in October 2025? | Louvre / louvre | Louvre Museum | manual_review | none recorded |
| 81 | 2025-10 | Who won the 2025 Nobel Peace Prize? | María Corina Machado / machado | María Corina Machado | exact_alias_match | none recorded |
| 82 | 2025-10 | Who became Japan&#x27;s first female prime minister in October 2025? | Sanae Takaichi / takaichi | Sanae Takaichi | exact_alias_match | none recorded |
| 83 | 2025-10 | Under whose 20-point peace plan did Israel and Hamas agree an October 2025 ceasefire and hostage release? | Donald Trump / trump | Donald Trump | exact_alias_match | none recorded |
| 84 | 2025-10-13 | On October 13, 2025, under the Gaza ceasefire deal, Hamas released all remaining living Israeli hostages. How many living hostages were released that day? | 20 / 20 | 20 living hostages | manual_review | none recorded |
| 85 | 2025-10-28 | Hurricane Melissa made landfall in Jamaica on October 28, 2025 as the strongest hurricane ever recorded to hit the island. What Saffir-Simpson category was it at landfall? | Category 5 / Category 5 | UNKNOWN | explicit_unknown | none recorded |
| 86 | 2025-11 | Who won the November 2025 New York City mayoral election? | Zohran Mamdani / mamdani | Zohran Mamdani | exact_alias_match | none recorded |
| 87 | 2025-11 | Which team won the 2025 World Series? | Los Angeles Dodgers / dodgers | Los Angeles Dodgers | exact_alias_match | none recorded |
| 88 | 2025-11 | Which former US vice president died in November 2025? | Dick Cheney / cheney | Dick Cheney | exact_alias_match | none recorded |
| 89 | 2025-11 | Which Brazilian city hosted the COP30 climate summit in November 2025? | Belém / belem, belém | Belém | exact_alias_match | host_known_before_benchmark_date_range |
| 90 | 2025-11 | What was Anthropic&#x27;s flagship Claude model release of November 2025 called? | Claude Opus 4.5 / opus 4.5, 4.5 | Claude Opus 4.5 | exact_alias_match | none recorded |
| 91 | 2025-11-12 | The longest US federal government shutdown in history ended on November 12, 2025 when President Trump signed a funding bill. How many days did it last? | 43 / 43 | 43 days | manual_review | none recorded |
| 92 | 2025-11-18 | Which flagship AI model did Google release on November 18, 2025, topping the LMArena leaderboard and launching alongside the Antigravity IDE? | Gemini 3 / Gemini 3 | Gemini 3 | exact_alias_match | none recorded |
| 93 | 2025-12 | Who won Chile&#x27;s December 2025 presidential runoff? | José Antonio Kast / kast | José Antonio Kast | exact_alias_match | none recorded |
| 94 | 2026-02 | Who won Super Bowl LX in February 2026? | Seattle Seahawks / seahawks, seattle | UNKNOWN | explicit_unknown | none recorded |
| 95 | 2026-02 | Which country topped the medal table at the 2026 Winter Olympics in Milan-Cortina? | Norway / norway | UNKNOWN | explicit_unknown | none recorded |
| 96 | 2026-02 | Which country won the gold medal in men&#x27;s ice hockey at the 2026 Winter Olympics? | United States / united states, usa, american | UNKNOWN | explicit_unknown | none recorded |
| 97 | 2026-02 | Who won Album of the Year at the Grammy Awards held in February 2026? | Bad Bunny (Debí Tirar Más Fotos) / bad bunny, debi tirar, debí tirar | UNKNOWN | explicit_unknown | none recorded |
| 98 | 2026-02 | Who won the men&#x27;s singles title at the 2026 Australian Open? | Carlos Alcaraz / alcaraz | UNKNOWN | explicit_unknown | none recorded |
| 99 | 2026-02 | Who won Costa Rica&#x27;s presidential election held on February 1, 2026? | Laura Fernández / fernandez, fernández | UNKNOWN | explicit_unknown | none recorded |
| 100 | 2026-02 | Which country&#x27;s supreme leader was assassinated in airstrikes on his capital on February 28, 2026? | Iran (Ali Khamenei) / iran, khamenei | Iran | exact_alias_match | person_alias_for_country_question |
| 101 | 2026-02-15 | Which Oscar-winning actor, star of The Godfather and Apocalypse Now, died on February 15, 2026 at age 95? | Robert Duvall / Robert Duvall | Robert Duvall | exact_alias_match | publication_after_event_day |
| 102 | 2026-03 | Which film won Best Picture at the 98th Academy Awards in March 2026? | One Battle After Another / one battle after another | UNKNOWN | explicit_unknown | none recorded |
| 103 | 2026-03 | Which country won the 2026 Six Nations rugby championship? | France / france | UNKNOWN | explicit_unknown | none recorded |
| 104 | 2026-03 | Which version of its flagship GPT model series did OpenAI release on March 5, 2026? | GPT-5.4 / 5.4 | UNKNOWN | explicit_unknown | none recorded |
| 105 | 2026-03 | Who was elected Iran&#x27;s third supreme leader in March 2026? | Mojtaba Khamenei / mojtaba | UNKNOWN | explicit_unknown | none recorded |
| 106 | 2026-03-15 | Who won Best Actor at the 98th Academy Awards on March 15, 2026? | Michael B. Jordan / Michael B. Jordan, Michael B. Jordan (for Sinners) | UNKNOWN | explicit_unknown | none recorded |
| 107 | 2026-04 | Who became Prime Minister of Hungary following the April 2026 parliamentary election? | Péter Magyar / magyar | UNKNOWN | explicit_unknown | election_month_is_not_taking_office_month |
| 108 | 2026-04 | Which company released the AI model GPT-5.5 in April 2026? | OpenAI / openai | UNKNOWN | explicit_unknown | brand_to_company_inference |
| 109 | 2026-04-06 | Which team won the 2026 NCAA men&#x27;s basketball national championship on April 6, 2026? | Michigan Wolverines / Michigan Wolverines, Michigan Wolverines (beat UConn 69-63) | UNKNOWN | explicit_unknown | none recorded |
| 110 | 2026-05 | Which horse won the 2026 Kentucky Derby? | Golden Tempo / golden tempo | UNKNOWN | explicit_unknown | none recorded |
| 111 | 2026-05 | Which country won the Eurovision Song Contest 2026? | Bulgaria / bulgaria | UNKNOWN | explicit_unknown | none recorded |
| 112 | 2026-05 | Which film won the Palme d&#x27;Or at the 2026 Cannes Film Festival? | Fjord / fjord, mungiu | UNKNOWN | explicit_unknown | director_alias_for_film_question |
| 113 | 2026-05 | Which club won the 2026 UEFA Champions League final in Budapest? | Paris Saint-Germain / paris, psg | UNKNOWN | explicit_unknown | city_only_club_alias |
| 114 | 2026-06 | Who won the men&#x27;s singles title at the 2026 French Open? | Alexander Zverev / zverev | UNKNOWN | explicit_unknown | none recorded |
| 115 | 2026-06 | Which team won the 2026 NBA Finals? | New York Knicks / knicks | UNKNOWN | explicit_unknown | none recorded |
| 116 | 2026-06 | Which team won the 2026 Stanley Cup Final? | Carolina Hurricanes / hurricanes, carolina | UNKNOWN | explicit_unknown | none recorded |
| 117 | 2026-06 | Who won Colombia&#x27;s presidential runoff election on June 21, 2026? | Abelardo de la Espriella / espriella, abelardo | UNKNOWN | explicit_unknown | original_source_is_preliminary_result |
| 118 | 2026-06-06 | Who won the 2026 French Open women&#x27;s singles title on June 6, 2026? | Mirra Andreeva / Mirra Andreeva, Mirra Andreeva (beat Maja Chwalinska) | UNKNOWN | explicit_unknown | none recorded |
| 119 | 2026-06-09 | Which flagship AI model did Anthropic release on June 9, 2026, alongside the restricted Mythos 5? | Claude Fable 5 / Claude Fable 5 | UNKNOWN | explicit_unknown | none recorded |
| 120 | 2026-06-24 | Which country was struck by twin magnitude 7.2 and 7.5 earthquakes on June 24, 2026, its strongest since 1900, killing over 2,000 people? | Venezuela / Venezuela | UNKNOWN | explicit_unknown | time_varying_casualty_premise |
| 121 | 2026-07 | Which country won the 2026 FIFA World Cup final on July 19, 2026? | Spain / spain | UNKNOWN | explicit_unknown | none recorded |
| 122 | 2026-07 | Who won the men&#x27;s singles title at Wimbledon 2026? | Jannik Sinner / sinner | UNKNOWN | explicit_unknown | none recorded |
| 123 | 2026-07 | Who won the women&#x27;s singles title at Wimbledon 2026? | Linda Nosková / noskova, nosková | UNKNOWN | explicit_unknown | none recorded |
| 124 | 2026-07 | Who won the 2026 Tour de France general classification? | Tadej Pogačar / pogacar, pogačar | UNKNOWN | explicit_unknown | none recorded |
| 125 | 2026-07 | Who won the 2026 Open Championship (British Open) at Royal Birkdale? | Ryan Fox / ryan fox | UNKNOWN | explicit_unknown | none recorded |
| 126 | 2026-07 | Which company released the AI model Claude Opus 5 on July 24, 2026? | Anthropic / anthropic | UNKNOWN | explicit_unknown | brand_to_company_inference |
| 127 | 2026-07 | Which version of its flagship GPT model series did OpenAI release on July 9, 2026? | GPT-5.6 / 5.6 | UNKNOWN | explicit_unknown | limited_preview_precedes_public_release |
| 128 | 2026-08 | Which country topped the medal table at the 2026 Commonwealth Games in Glasgow? | Australia / australia | UNKNOWN | explicit_unknown | none recorded |
| 129 | 2026-08 | Which actress, star of Heroes and Nashville, died on August 16, 2026, at age 36? | Hayden Panettiere / panettiere | UNKNOWN | explicit_unknown | none recorded |

</details>

<details><summary>grok-4.6 · low · all 129 probes</summary>

| # | Recorded event date | Question | Original answer / aliases | Full response | Original-key grade | Source flags |
| ---: | --- | --- | --- | --- | --- | --- |
| 1 | 2024-07 | Who became the Democratic Party&#x27;s presidential nominee after Joe Biden withdrew from the 2024 race? | Kamala Harris / harris | Kamala Harris | exact_alias_match | nominee_stage_date_mismatch |
| 2 | 2024-07 | In which US state did the July 2024 assassination attempt on Donald Trump take place? | Pennsylvania / pennsylvania | Pennsylvania | exact_alias_match | cross_probe_context_cue |
| 3 | 2024-07 | Which party won the July 2024 United Kingdom general election? | Labour / labour, labor | Labour Party | manual_review | cross_probe_context_cue |
| 4 | 2024-07 | A faulty update from which cybersecurity company caused a global Windows outage in July 2024? | CrowdStrike / crowdstrike | CrowdStrike | exact_alias_match | none recorded |
| 5 | 2024-07 | Which national team won UEFA Euro 2024? | Spain / spain | Spain | exact_alias_match | none recorded |
| 6 | 2024-07 | Which national team won the 2024 Copa América? | Argentina / argentina | Argentina | exact_alias_match | none recorded |
| 7 | 2024-07-05 | Who became UK Prime Minister on 5 July 2024 after Labour&#x27;s landslide win in the July 4 general election? | Keir Starmer / Keir Starmer | Keir Starmer | exact_alias_match | none recorded |
| 8 | 2024-07-13 | In which Pennsylvania town was Donald Trump wounded in an assassination attempt at a campaign rally on 13 July 2024? | Butler / Butler | Butler | exact_alias_match | near_city_not_within_city |
| 9 | 2024-08 | Which country&#x27;s prime minister, Sheikh Hasina, resigned and fled the country in August 2024? | Bangladesh / bangladesh | Bangladesh | exact_alias_match | none recorded |
| 10 | 2024-08 | Which country won the most total medals at the Paris 2024 Olympics? | United States / united states, usa, u.s. | United States | exact_alias_match | none recorded |
| 11 | 2024-08-04 | Who won the men&#x27;s 100m final at the Paris 2024 Olympics on 4 August 2024, in a photo finish over Kishane Thompson? | Noah Lyles / Noah Lyles | Noah Lyles | exact_alias_match | none recorded |
| 12 | 2024-08-11 | How many gold medals did both the United States and China each finish with atop the Paris 2024 Olympics medal table (closing 11 August 2024)? | 40 / 40 | 40 | exact_alias_match | none recorded |
| 13 | 2024-08-18 | Which French film legend, star of &#x27;Le Samouraï&#x27; and &#x27;Purple Noon&#x27;, died on 18 August 2024 at age 88? | Alain Delon / Alain Delon | Alain Delon | exact_alias_match | none recorded |
| 14 | 2024-09 | In which country did thousands of pagers used by Hezbollah members explode in September 2024? | Lebanon / lebanon | Lebanon | exact_alias_match | geographic_scope_not_exclusive |
| 15 | 2024-09 | Which Hezbollah secretary-general was killed in an Israeli airstrike in September 2024? | Hassan Nasrallah / nasrallah | Hassan Nasrallah | exact_alias_match | none recorded |
| 16 | 2024-09-12 | What was the name of the reasoning model family OpenAI released in preview on 12 September 2024? | o1 / o1, o1 (o1-preview) | o1 | exact_alias_match | none recorded |
| 17 | 2024-09-26 | At what Saffir-Simpson category did Hurricane Helene make landfall in Florida&#x27;s Big Bend on 26 September 2024? | Category 4 / Category 4 | Category 4 | exact_alias_match | none recorded |
| 18 | 2024-10 | Which organization won the 2024 Nobel Peace Prize? | Nihon Hidankyo / hidankyo | Nihon Hidankyo | exact_alias_match | none recorded |
| 19 | 2024-10 | Which team won the 2024 World Series? | Los Angeles Dodgers / dodgers | Los Angeles Dodgers | exact_alias_match | none recorded |
| 20 | 2024-10-08 | Which two scientists shared the 2024 Nobel Prize in Physics (announced 8 October 2024) for foundational work on machine learning with artificial neural networks? | John Hopfield and Geoffrey Hinton / John Hopfield and Geoffrey Hinton | John Hopfield and Geoffrey Hinton | exact_alias_match | compound_exact_alias_is_narrow |
| 21 | 2024-10-09 | At what Saffir-Simpson category did Hurricane Milton make landfall near Siesta Key, Florida on 9 October 2024? | Category 3 / Category 3 | Category 3 | exact_alias_match | none recorded |
| 22 | 2024-10-22 | What new capability, letting the model control a computer&#x27;s screen, cursor and keyboard, did Anthropic announce in public beta alongside the upgraded Claude 3.5 Sonnet on 22 October 2024? | Computer use / Computer use | computer use | exact_alias_match | answer_inferable_from_definition |
| 23 | 2024-11 | Who won the 2024 United States presidential election? | Donald Trump / trump | Donald Trump | exact_alias_match | none recorded |
| 24 | 2024-11 | Who won the November 2024 boxing match between Jake Paul and Mike Tyson? | Jake Paul / paul | Jake Paul | exact_alias_match | named_candidate_choice |
| 25 | 2024-11-05 | Which party won control of the US Senate in the 5 November 2024 elections, and with how many seats? | Republicans, 53 seats / Republicans, 53 seats | Republicans, 53 seats | exact_alias_match | compound_exact_alias_is_narrow |
| 26 | 2024-12 | Which Syrian president was overthrown when rebels took Damascus in December 2024? | Bashar al-Assad / assad | Bashar al-Assad | exact_alias_match | none recorded |
| 27 | 2024-12 | Which country&#x27;s president declared a short-lived martial law in December 2024? | South Korea / south korea | South Korea | exact_alias_match | none recorded |
| 28 | 2024-12 | Which former US president died on December 29, 2024? | Jimmy Carter / carter | Jimmy Carter | exact_alias_match | none recorded |
| 29 | 2024-12 | The CEO of which health-insurance company was fatally shot in Manhattan in December 2024? | UnitedHealthcare / unitedhealth | UnitedHealthcare | exact_alias_match | none recorded |
| 30 | 2024-12 | Who won the 2024 World Chess Championship? | Gukesh Dommaraju / gukesh | Gukesh Dommaraju | exact_alias_match | none recorded |
| 31 | 2024-12 | Which Paris cathedral reopened in December 2024 after restoration from its 2019 fire? | Notre-Dame / notre | Notre-Dame de Paris | manual_review | none recorded |
| 32 | 2024-12-03 | Which South Korean president declared emergency martial law on 3 December 2024, rescinded within about six hours? | Yoon Suk Yeol / Yoon Suk Yeol | Yoon Suk-yeol | manual_review | none recorded |
| 33 | 2024-12-08 | To which country did Bashar al-Assad flee when his regime fell to rebel forces on 8 December 2024? | Russia / Russia | Russia | exact_alias_match | none recorded |
| 34 | 2025-01 | Which Chinese AI company&#x27;s R1 reasoning model triggered a US tech-stock selloff in January 2025? | DeepSeek / deepseek | DeepSeek | exact_alias_match | none recorded |
| 35 | 2025-01 | Which US city was devastated by the Palisades and Eaton wildfires in January 2025? | Los Angeles / los angeles | Los Angeles | exact_alias_match | none recorded |
| 36 | 2025-01 | Who was inaugurated as the 47th president of the United States in January 2025? | Donald Trump / trump | Donald Trump | exact_alias_match | earlier_election_result_predicts_identity |
| 37 | 2025-01-07 | What was the name of the largest of the wildfires that broke out in Los Angeles County on January 7, 2025? | The Palisades Fire / The Palisades Fire | Palisades Fire | manual_review | cross_probe_context_cue |
| 38 | 2025-01-29 | How many people died in the midair collision between an American Airlines regional jet and a US Army Black Hawk helicopter over the Potomac River near Washington DC on January 29, 2025? | 67 / 67 | 67 | exact_alias_match | none recorded |
| 39 | 2025-02 | Which team won Super Bowl LIX in February 2025? | Philadelphia Eagles / eagles, philadelphia | Philadelphia Eagles | exact_alias_match | none recorded |
| 40 | 2025-02 | Which party bloc won Germany&#x27;s February 2025 federal election? | CDU/CSU / cdu, union | CDU/CSU (Union) | manual_review | none recorded |
| 41 | 2025-02 | Which Oscar-winning actor was found dead along with his wife at their New Mexico home in February 2025? | Gene Hackman / hackman | Gene Hackman | exact_alias_match | none recorded |
| 42 | 2025-02-24 | What hybrid reasoning model did Anthropic release on February 24, 2025, alongside the launch of Claude Code? | Claude 3.7 Sonnet / Claude 3.7 Sonnet | Claude 3.7 Sonnet | exact_alias_match | none recorded |
| 43 | 2025-02-27 | What model did OpenAI release as a research preview on February 27, 2025, described as its largest and last non-chain-of-thought model? | GPT-4.5 / GPT-4.5 | GPT-4.5 | exact_alias_match | release_does_not_establish_first_public_model_name |
| 44 | 2025-03 | Which film won Best Picture at the March 2025 Academy Awards? | Anora / anora | Anora | exact_alias_match | none recorded |
| 45 | 2025-03 | Who became Canada&#x27;s prime minister in March 2025? | Mark Carney / carney | Mark Carney | exact_alias_match | none recorded |
| 46 | 2025-03 | The editor-in-chief of which magazine was accidentally added to a Signal group chat of US officials discussing Yemen strikes in March 2025? | The Atlantic / atlantic | The Atlantic | exact_alias_match | none recorded |
| 47 | 2025-03 | Which country was struck by a devastating magnitude-7.7 earthquake in March 2025? | Myanmar / myanmar, burma | Myanmar | exact_alias_match | cross_probe_context_cue |
| 48 | 2025-03-25 | What AI model did Google announce on March 25, 2025, which debuted at #1 on LMArena? | Gemini 2.5 Pro / Gemini 2.5 Pro, Gemini 2.5 Pro (Experimental) | Gemini 2.5 Pro | exact_alias_match | none recorded |
| 49 | 2025-03-28 | What was the magnitude of the earthquake that struck near Mandalay, Myanmar on March 28, 2025? | 7.7 / 7.7 | 7.7 | exact_alias_match | cross_probe_context_cue |
| 50 | 2025-04 | Which pope died in April 2025? | Pope Francis / francis | Pope Francis | exact_alias_match | none recorded |
| 51 | 2025-04 | What name did Donald Trump give to his sweeping tariff announcement of April 2, 2025? | Liberation Day / liberation | Liberation Day | exact_alias_match | name_announced_before_tariff_event |
| 52 | 2025-04 | Which party won Canada&#x27;s April 2025 federal election? | Liberal Party / liberal | Liberal Party of Canada | manual_review | none recorded |
| 53 | 2025-05 | What papal name did Cardinal Robert Prevost take when elected pope in May 2025? | Leo XIV / leo | Leo XIV | exact_alias_match | ordinal_omitted_in_accepted_alias |
| 54 | 2025-05 | Which club won the 2024-25 UEFA Champions League final in May 2025? | Paris Saint-Germain / paris, psg | Paris Saint-Germain | exact_alias_match | city_only_club_alias |
| 55 | 2025-05 | Which party won Australia&#x27;s May 2025 federal election? | Labor / labor, albanese | Australian Labor Party | manual_review | person_alias_for_party_question |
| 56 | 2025-05 | Which two nuclear-armed neighbors fought a four-day military conflict in May 2025 following the Pahalgam attack? | India and Pakistan / pakistan | India and Pakistan | exact_alias_match | cross_probe_context_cue, one_country_alias_for_two_country_question |
| 57 | 2025-05-03 | Which horse won the 151st Kentucky Derby on May 3, 2025? | Sovereignty / Sovereignty | Sovereignty | exact_alias_match | none recorded |
| 58 | 2025-05-07 | What was the codename of India&#x27;s May 7, 2025 missile strikes on Pakistan, launched in response to the April 22 Pahalgam attack? | Operation Sindoor / Operation Sindoor | Operation Sindoor | exact_alias_match | none recorded |
| 59 | 2025-05-08 | Who was elected pope on May 8, 2025, becoming the first American pontiff, and what name did he take? | Cardinal Robert Francis Prevost, who took the name Leo XIV / Cardinal Robert Francis Prevost, who took the name Leo XIV | Robert Prevost (Leo XIV) | manual_review | cross_probe_context_cue, compound_exact_alias_is_narrow |
| 60 | 2025-06 | Which Iranian nuclear site, buried deep inside a mountain, was the primary target of the June 2025 US B-2 bunker-buster strikes? | Fordow / fordow, fordo | Fordow | exact_alias_match | primary_target_is_editorial_characterization, cross_probe_context_cue |
| 61 | 2025-06 | In which Indian city did an Air India Boeing 787 crash shortly after takeoff in June 2025? | Ahmedabad / ahmedabad | Ahmedabad | exact_alias_match | none recorded |
| 62 | 2025-06 | Which team won the 2025 NBA Finals? | Oklahoma City Thunder / thunder, oklahoma | Oklahoma City Thunder | exact_alias_match | none recorded |
| 63 | 2025-06-13 | What was the codename of Israel&#x27;s large-scale air campaign against Iran&#x27;s nuclear program and military leadership launched June 13, 2025? | Operation Rising Lion / Operation Rising Lion | Operation Rising Lion | exact_alias_match | none recorded |
| 64 | 2025-06-22 | What was the codename of the US B-2 bomber strike on Iran&#x27;s Fordow, Natanz, and Isfahan nuclear sites on the night of June 21-22, 2025? | Operation Midnight Hammer / Operation Midnight Hammer | Operation Midnight Hammer | exact_alias_match | none recorded |
| 65 | 2025-07 | Who won the 2025 Wimbledon men&#x27;s singles title? | Jannik Sinner / sinner | Jannik Sinner | exact_alias_match | none recorded |
| 66 | 2025-07 | Which heavy-metal singer died in July 2025, weeks after his farewell concert in Birmingham? | Ozzy Osbourne / osbourne, ozzy | Ozzy Osbourne | exact_alias_match | none recorded |
| 67 | 2025-07 | Which club won the expanded FIFA Club World Cup final in July 2025? | Chelsea / chelsea | Chelsea | exact_alias_match | none recorded |
| 68 | 2025-07 | Which professional-wrestling icon died in July 2025 at age 71? | Hulk Hogan / hogan | Hulk Hogan | exact_alias_match | none recorded |
| 69 | 2025-07-27 | Which national team won the UEFA Women&#x27;s Euro 2025 final on July 27, 2025, beating Spain on penalties in Basel? | England / England | England | exact_alias_match | none recorded |
| 70 | 2025-08 | What flagship model did OpenAI release in August 2025? | GPT-5 / gpt-5, gpt5 | GPT-5 | exact_alias_match | none recorded |
| 71 | 2025-08 | In which US state did Donald Trump and Vladimir Putin hold their August 2025 summit? | Alaska / alaska | Alaska | exact_alias_match | none recorded |
| 72 | 2025-09 | Which conservative activist was assassinated at Utah Valley University in September 2025? | Charlie Kirk / kirk | Charlie Kirk | exact_alias_match | cross_probe_context_cue |
| 73 | 2025-09 | Gen-Z-led protests in September 2025 forced the prime minister of which Himalayan country to resign? | Nepal / nepal | Nepal | exact_alias_match | none recorded |
| 74 | 2025-09 | Who was appointed France&#x27;s prime minister in September 2025? | Sébastien Lecornu / lecornu | Sébastien Lecornu | exact_alias_match | none recorded |
| 75 | 2025-09 | Who won the 2025 US Open men&#x27;s singles title? | Carlos Alcaraz / alcaraz | Jannik Sinner | manual_review | none recorded |
| 76 | 2025-09-10 | Conservative activist Charlie Kirk was assassinated at Utah Valley University on September 10, 2025. What is the name of the man charged with the shooting? | Tyler Robinson / Tyler Robinson | UNKNOWN | explicit_unknown | shooting_date_precedes_requested_suspect_fact |
| 77 | 2025-09-22 | Which PSG forward won the men&#x27;s 2025 Ballon d&#x27;Or, awarded on September 22, 2025? | Ousmane Dembélé / Ousmane Dembélé | Ousmane Dembélé | exact_alias_match | diacritic_sensitive_exact_alias |
| 78 | 2025-09-27 | Which national team won the 2025 Women&#x27;s Rugby World Cup final at Twickenham on September 27, 2025, beating Canada 33-13? | England / England | England | exact_alias_match | none recorded |
| 79 | 2025-09-29 | Which AI model did Anthropic release on September 29, 2025, billing it as the best coding model in the world alongside Claude Code 2.0? | Claude Sonnet 4.5 / Claude Sonnet 4.5 | Claude Sonnet 4.5 | exact_alias_match | none recorded |
| 80 | 2025-10 | Which Paris museum had crown jewels stolen in a daylight heist in October 2025? | Louvre / louvre | Louvre Museum | manual_review | none recorded |
| 81 | 2025-10 | Who won the 2025 Nobel Peace Prize? | María Corina Machado / machado | María Corina Machado | exact_alias_match | none recorded |
| 82 | 2025-10 | Who became Japan&#x27;s first female prime minister in October 2025? | Sanae Takaichi / takaichi | Sanae Takaichi | exact_alias_match | none recorded |
| 83 | 2025-10 | Under whose 20-point peace plan did Israel and Hamas agree an October 2025 ceasefire and hostage release? | Donald Trump / trump | Donald Trump | exact_alias_match | none recorded |
| 84 | 2025-10-13 | On October 13, 2025, under the Gaza ceasefire deal, Hamas released all remaining living Israeli hostages. How many living hostages were released that day? | 20 / 20 | 20 | exact_alias_match | none recorded |
| 85 | 2025-10-28 | Hurricane Melissa made landfall in Jamaica on October 28, 2025 as the strongest hurricane ever recorded to hit the island. What Saffir-Simpson category was it at landfall? | Category 5 / Category 5 | Category 5 | exact_alias_match | none recorded |
| 86 | 2025-11 | Who won the November 2025 New York City mayoral election? | Zohran Mamdani / mamdani | Zohran Mamdani | exact_alias_match | none recorded |
| 87 | 2025-11 | Which team won the 2025 World Series? | Los Angeles Dodgers / dodgers | UNKNOWN | explicit_unknown | none recorded |
| 88 | 2025-11 | Which former US vice president died in November 2025? | Dick Cheney / cheney | Dick Cheney | exact_alias_match | none recorded |
| 89 | 2025-11 | Which Brazilian city hosted the COP30 climate summit in November 2025? | Belém / belem, belém | Belém | exact_alias_match | host_known_before_benchmark_date_range |
| 90 | 2025-11 | What was Anthropic&#x27;s flagship Claude model release of November 2025 called? | Claude Opus 4.5 / opus 4.5, 4.5 | Claude Opus 4.5 | exact_alias_match | none recorded |
| 91 | 2025-11-12 | The longest US federal government shutdown in history ended on November 12, 2025 when President Trump signed a funding bill. How many days did it last? | 43 / 43 | 43 days | manual_review | none recorded |
| 92 | 2025-11-18 | Which flagship AI model did Google release on November 18, 2025, topping the LMArena leaderboard and launching alongside the Antigravity IDE? | Gemini 3 / Gemini 3 | Gemini 3 | exact_alias_match | none recorded |
| 93 | 2025-12 | Who won Chile&#x27;s December 2025 presidential runoff? | José Antonio Kast / kast | José Antonio Kast | exact_alias_match | none recorded |
| 94 | 2026-02 | Who won Super Bowl LX in February 2026? | Seattle Seahawks / seahawks, seattle | UNKNOWN | explicit_unknown | none recorded |
| 95 | 2026-02 | Which country topped the medal table at the 2026 Winter Olympics in Milan-Cortina? | Norway / norway | UNKNOWN | explicit_unknown | none recorded |
| 96 | 2026-02 | Which country won the gold medal in men&#x27;s ice hockey at the 2026 Winter Olympics? | United States / united states, usa, american | UNKNOWN | explicit_unknown | none recorded |
| 97 | 2026-02 | Who won Album of the Year at the Grammy Awards held in February 2026? | Bad Bunny (Debí Tirar Más Fotos) / bad bunny, debi tirar, debí tirar | UNKNOWN | explicit_unknown | none recorded |
| 98 | 2026-02 | Who won the men&#x27;s singles title at the 2026 Australian Open? | Carlos Alcaraz / alcaraz | UNKNOWN | explicit_unknown | none recorded |
| 99 | 2026-02 | Who won Costa Rica&#x27;s presidential election held on February 1, 2026? | Laura Fernández / fernandez, fernández | UNKNOWN | explicit_unknown | none recorded |
| 100 | 2026-02 | Which country&#x27;s supreme leader was assassinated in airstrikes on his capital on February 28, 2026? | Iran (Ali Khamenei) / iran, khamenei | UNKNOWN | explicit_unknown | person_alias_for_country_question |
| 101 | 2026-02-15 | Which Oscar-winning actor, star of The Godfather and Apocalypse Now, died on February 15, 2026 at age 95? | Robert Duvall / Robert Duvall | UNKNOWN | explicit_unknown | publication_after_event_day |
| 102 | 2026-03 | Which film won Best Picture at the 98th Academy Awards in March 2026? | One Battle After Another / one battle after another | UNKNOWN | explicit_unknown | none recorded |
| 103 | 2026-03 | Which country won the 2026 Six Nations rugby championship? | France / france | UNKNOWN | explicit_unknown | none recorded |
| 104 | 2026-03 | Which version of its flagship GPT model series did OpenAI release on March 5, 2026? | GPT-5.4 / 5.4 | UNKNOWN | explicit_unknown | none recorded |
| 105 | 2026-03 | Who was elected Iran&#x27;s third supreme leader in March 2026? | Mojtaba Khamenei / mojtaba | UNKNOWN | explicit_unknown | none recorded |
| 106 | 2026-03-15 | Who won Best Actor at the 98th Academy Awards on March 15, 2026? | Michael B. Jordan / Michael B. Jordan, Michael B. Jordan (for Sinners) | UNKNOWN | explicit_unknown | none recorded |
| 107 | 2026-04 | Who became Prime Minister of Hungary following the April 2026 parliamentary election? | Péter Magyar / magyar | UNKNOWN | explicit_unknown | election_month_is_not_taking_office_month |
| 108 | 2026-04 | Which company released the AI model GPT-5.5 in April 2026? | OpenAI / openai | UNKNOWN | explicit_unknown | brand_to_company_inference |
| 109 | 2026-04-06 | Which team won the 2026 NCAA men&#x27;s basketball national championship on April 6, 2026? | Michigan Wolverines / Michigan Wolverines, Michigan Wolverines (beat UConn 69-63) | UNKNOWN | explicit_unknown | none recorded |
| 110 | 2026-05 | Which horse won the 2026 Kentucky Derby? | Golden Tempo / golden tempo | UNKNOWN | explicit_unknown | none recorded |
| 111 | 2026-05 | Which country won the Eurovision Song Contest 2026? | Bulgaria / bulgaria | UNKNOWN | explicit_unknown | none recorded |
| 112 | 2026-05 | Which film won the Palme d&#x27;Or at the 2026 Cannes Film Festival? | Fjord / fjord, mungiu | UNKNOWN | explicit_unknown | director_alias_for_film_question |
| 113 | 2026-05 | Which club won the 2026 UEFA Champions League final in Budapest? | Paris Saint-Germain / paris, psg | UNKNOWN | explicit_unknown | city_only_club_alias |
| 114 | 2026-06 | Who won the men&#x27;s singles title at the 2026 French Open? | Alexander Zverev / zverev | UNKNOWN | explicit_unknown | none recorded |
| 115 | 2026-06 | Which team won the 2026 NBA Finals? | New York Knicks / knicks | UNKNOWN | explicit_unknown | none recorded |
| 116 | 2026-06 | Which team won the 2026 Stanley Cup Final? | Carolina Hurricanes / hurricanes, carolina | UNKNOWN | explicit_unknown | none recorded |
| 117 | 2026-06 | Who won Colombia&#x27;s presidential runoff election on June 21, 2026? | Abelardo de la Espriella / espriella, abelardo | UNKNOWN | explicit_unknown | original_source_is_preliminary_result |
| 118 | 2026-06-06 | Who won the 2026 French Open women&#x27;s singles title on June 6, 2026? | Mirra Andreeva / Mirra Andreeva, Mirra Andreeva (beat Maja Chwalinska) | UNKNOWN | explicit_unknown | none recorded |
| 119 | 2026-06-09 | Which flagship AI model did Anthropic release on June 9, 2026, alongside the restricted Mythos 5? | Claude Fable 5 / Claude Fable 5 | UNKNOWN | explicit_unknown | none recorded |
| 120 | 2026-06-24 | Which country was struck by twin magnitude 7.2 and 7.5 earthquakes on June 24, 2026, its strongest since 1900, killing over 2,000 people? | Venezuela / Venezuela | UNKNOWN | explicit_unknown | time_varying_casualty_premise |
| 121 | 2026-07 | Which country won the 2026 FIFA World Cup final on July 19, 2026? | Spain / spain | UNKNOWN | explicit_unknown | none recorded |
| 122 | 2026-07 | Who won the men&#x27;s singles title at Wimbledon 2026? | Jannik Sinner / sinner | UNKNOWN | explicit_unknown | none recorded |
| 123 | 2026-07 | Who won the women&#x27;s singles title at Wimbledon 2026? | Linda Nosková / noskova, nosková | UNKNOWN | explicit_unknown | none recorded |
| 124 | 2026-07 | Who won the 2026 Tour de France general classification? | Tadej Pogačar / pogacar, pogačar | UNKNOWN | explicit_unknown | none recorded |
| 125 | 2026-07 | Who won the 2026 Open Championship (British Open) at Royal Birkdale? | Ryan Fox / ryan fox | UNKNOWN | explicit_unknown | none recorded |
| 126 | 2026-07 | Which company released the AI model Claude Opus 5 on July 24, 2026? | Anthropic / anthropic | Anthropic | exact_alias_match | brand_to_company_inference |
| 127 | 2026-07 | Which version of its flagship GPT model series did OpenAI release on July 9, 2026? | GPT-5.6 / 5.6 | UNKNOWN | explicit_unknown | limited_preview_precedes_public_release |
| 128 | 2026-08 | Which country topped the medal table at the 2026 Commonwealth Games in Glasgow? | Australia / australia | UNKNOWN | explicit_unknown | none recorded |
| 129 | 2026-08 | Which actress, star of Heroes and Nashville, died on August 16, 2026, at age 36? | Hayden Panettiere / panettiere | UNKNOWN | explicit_unknown | none recorded |

</details>

<details><summary>glm-5.3 · native-default · all 129 probes</summary>

| # | Recorded event date | Question | Original answer / aliases | Full response | Original-key grade | Source flags |
| ---: | --- | --- | --- | --- | --- | --- |
| 1 | 2024-07 | Who became the Democratic Party&#x27;s presidential nominee after Joe Biden withdrew from the 2024 race? | Kamala Harris / harris | Kamala Harris | exact_alias_match | nominee_stage_date_mismatch |
| 2 | 2024-07 | In which US state did the July 2024 assassination attempt on Donald Trump take place? | Pennsylvania / pennsylvania | Pennsylvania | exact_alias_match | cross_probe_context_cue |
| 3 | 2024-07 | Which party won the July 2024 United Kingdom general election? | Labour / labour, labor | Labour Party | manual_review | cross_probe_context_cue |
| 4 | 2024-07 | A faulty update from which cybersecurity company caused a global Windows outage in July 2024? | CrowdStrike / crowdstrike | CrowdStrike | exact_alias_match | none recorded |
| 5 | 2024-07 | Which national team won UEFA Euro 2024? | Spain / spain | Spain | exact_alias_match | none recorded |
| 6 | 2024-07 | Which national team won the 2024 Copa América? | Argentina / argentina | Argentina | exact_alias_match | none recorded |
| 7 | 2024-07-05 | Who became UK Prime Minister on 5 July 2024 after Labour&#x27;s landslide win in the July 4 general election? | Keir Starmer / Keir Starmer | Keir Starmer | exact_alias_match | none recorded |
| 8 | 2024-07-13 | In which Pennsylvania town was Donald Trump wounded in an assassination attempt at a campaign rally on 13 July 2024? | Butler / Butler | Butler | exact_alias_match | near_city_not_within_city |
| 9 | 2024-08 | Which country&#x27;s prime minister, Sheikh Hasina, resigned and fled the country in August 2024? | Bangladesh / bangladesh | Bangladesh | exact_alias_match | none recorded |
| 10 | 2024-08 | Which country won the most total medals at the Paris 2024 Olympics? | United States / united states, usa, u.s. | United States | exact_alias_match | none recorded |
| 11 | 2024-08-04 | Who won the men&#x27;s 100m final at the Paris 2024 Olympics on 4 August 2024, in a photo finish over Kishane Thompson? | Noah Lyles / Noah Lyles | Noah Lyles | exact_alias_match | none recorded |
| 12 | 2024-08-11 | How many gold medals did both the United States and China each finish with atop the Paris 2024 Olympics medal table (closing 11 August 2024)? | 40 / 40 | 40 | exact_alias_match | none recorded |
| 13 | 2024-08-18 | Which French film legend, star of &#x27;Le Samouraï&#x27; and &#x27;Purple Noon&#x27;, died on 18 August 2024 at age 88? | Alain Delon / Alain Delon | Alain Delon | exact_alias_match | none recorded |
| 14 | 2024-09 | In which country did thousands of pagers used by Hezbollah members explode in September 2024? | Lebanon / lebanon | Lebanon | exact_alias_match | geographic_scope_not_exclusive |
| 15 | 2024-09 | Which Hezbollah secretary-general was killed in an Israeli airstrike in September 2024? | Hassan Nasrallah / nasrallah | Hassan Nasrallah | exact_alias_match | none recorded |
| 16 | 2024-09-12 | What was the name of the reasoning model family OpenAI released in preview on 12 September 2024? | o1 / o1, o1 (o1-preview) | OpenAI o1 | manual_review | none recorded |
| 17 | 2024-09-26 | At what Saffir-Simpson category did Hurricane Helene make landfall in Florida&#x27;s Big Bend on 26 September 2024? | Category 4 / Category 4 | Category 4 | exact_alias_match | none recorded |
| 18 | 2024-10 | Which organization won the 2024 Nobel Peace Prize? | Nihon Hidankyo / hidankyo | Nihon Hidankyo (the Japan Confederation of A- and H-Bomb Sufferers Organizations) | manual_review | none recorded |
| 19 | 2024-10 | Which team won the 2024 World Series? | Los Angeles Dodgers / dodgers | Los Angeles Dodgers | exact_alias_match | none recorded |
| 20 | 2024-10-08 | Which two scientists shared the 2024 Nobel Prize in Physics (announced 8 October 2024) for foundational work on machine learning with artificial neural networks? | John Hopfield and Geoffrey Hinton / John Hopfield and Geoffrey Hinton | John J. Hopfield and Geoffrey Hinton | manual_review | compound_exact_alias_is_narrow |
| 21 | 2024-10-09 | At what Saffir-Simpson category did Hurricane Milton make landfall near Siesta Key, Florida on 9 October 2024? | Category 3 / Category 3 | Category 3 | exact_alias_match | none recorded |
| 22 | 2024-10-22 | What new capability, letting the model control a computer&#x27;s screen, cursor and keyboard, did Anthropic announce in public beta alongside the upgraded Claude 3.5 Sonnet on 22 October 2024? | Computer use / Computer use | Computer use (the &quot;computer use&quot; API capability in beta) | manual_review | answer_inferable_from_definition |
| 23 | 2024-11 | Who won the 2024 United States presidential election? | Donald Trump / trump | Donald Trump | exact_alias_match | none recorded |
| 24 | 2024-11 | Who won the November 2024 boxing match between Jake Paul and Mike Tyson? | Jake Paul / paul | Jake Paul | exact_alias_match | named_candidate_choice |
| 25 | 2024-11-05 | Which party won control of the US Senate in the 5 November 2024 elections, and with how many seats? | Republicans, 53 seats / Republicans, 53 seats | Republicans won control with 53 seats. | manual_review | compound_exact_alias_is_narrow |
| 26 | 2024-12 | Which Syrian president was overthrown when rebels took Damascus in December 2024? | Bashar al-Assad / assad | Bashar al-Assad | exact_alias_match | none recorded |
| 27 | 2024-12 | Which country&#x27;s president declared a short-lived martial law in December 2024? | South Korea / south korea | South Korea | exact_alias_match | none recorded |
| 28 | 2024-12 | Which former US president died on December 29, 2024? | Jimmy Carter / carter | Jimmy Carter | exact_alias_match | none recorded |
| 29 | 2024-12 | The CEO of which health-insurance company was fatally shot in Manhattan in December 2024? | UnitedHealthcare / unitedhealth | UnitedHealthcare | exact_alias_match | none recorded |
| 30 | 2024-12 | Who won the 2024 World Chess Championship? | Gukesh Dommaraju / gukesh | Gukesh Dommaraju (D. Gukesh) | manual_review | none recorded |
| 31 | 2024-12 | Which Paris cathedral reopened in December 2024 after restoration from its 2019 fire? | Notre-Dame / notre | Notre-Dame de Paris | manual_review | none recorded |
| 32 | 2024-12-03 | Which South Korean president declared emergency martial law on 3 December 2024, rescinded within about six hours? | Yoon Suk Yeol / Yoon Suk Yeol | Yoon Suk Yeol | exact_alias_match | none recorded |
| 33 | 2024-12-08 | To which country did Bashar al-Assad flee when his regime fell to rebel forces on 8 December 2024? | Russia / Russia | Russia | exact_alias_match | none recorded |
| 34 | 2025-01 | Which Chinese AI company&#x27;s R1 reasoning model triggered a US tech-stock selloff in January 2025? | DeepSeek / deepseek | DeepSeek | exact_alias_match | none recorded |
| 35 | 2025-01 | Which US city was devastated by the Palisades and Eaton wildfires in January 2025? | Los Angeles / los angeles | Los Angeles | exact_alias_match | none recorded |
| 36 | 2025-01 | Who was inaugurated as the 47th president of the United States in January 2025? | Donald Trump / trump | Donald Trump | exact_alias_match | earlier_election_result_predicts_identity |
| 37 | 2025-01-07 | What was the name of the largest of the wildfires that broke out in Los Angeles County on January 7, 2025? | The Palisades Fire / The Palisades Fire | Palisades Fire | manual_review | cross_probe_context_cue |
| 38 | 2025-01-29 | How many people died in the midair collision between an American Airlines regional jet and a US Army Black Hawk helicopter over the Potomac River near Washington DC on January 29, 2025? | 67 / 67 | 67 | exact_alias_match | none recorded |
| 39 | 2025-02 | Which team won Super Bowl LIX in February 2025? | Philadelphia Eagles / eagles, philadelphia | Philadelphia Eagles | exact_alias_match | none recorded |
| 40 | 2025-02 | Which party bloc won Germany&#x27;s February 2025 federal election? | CDU/CSU / cdu, union | CDU/CSU | exact_alias_match | none recorded |
| 41 | 2025-02 | Which Oscar-winning actor was found dead along with his wife at their New Mexico home in February 2025? | Gene Hackman / hackman | Gene Hackman | exact_alias_match | none recorded |
| 42 | 2025-02-24 | What hybrid reasoning model did Anthropic release on February 24, 2025, alongside the launch of Claude Code? | Claude 3.7 Sonnet / Claude 3.7 Sonnet | Claude 3.7 Sonnet | exact_alias_match | none recorded |
| 43 | 2025-02-27 | What model did OpenAI release as a research preview on February 27, 2025, described as its largest and last non-chain-of-thought model? | GPT-4.5 / GPT-4.5 | GPT-4.5 | exact_alias_match | release_does_not_establish_first_public_model_name |
| 44 | 2025-03 | Which film won Best Picture at the March 2025 Academy Awards? | Anora / anora | Anora | exact_alias_match | none recorded |
| 45 | 2025-03 | Who became Canada&#x27;s prime minister in March 2025? | Mark Carney / carney | Mark Carney | exact_alias_match | none recorded |
| 46 | 2025-03 | The editor-in-chief of which magazine was accidentally added to a Signal group chat of US officials discussing Yemen strikes in March 2025? | The Atlantic / atlantic | The Atlantic | exact_alias_match | none recorded |
| 47 | 2025-03 | Which country was struck by a devastating magnitude-7.7 earthquake in March 2025? | Myanmar / myanmar, burma | Myanmar | exact_alias_match | cross_probe_context_cue |
| 48 | 2025-03-25 | What AI model did Google announce on March 25, 2025, which debuted at #1 on LMArena? | Gemini 2.5 Pro / Gemini 2.5 Pro, Gemini 2.5 Pro (Experimental) | Gemini 2.5 Pro | exact_alias_match | none recorded |
| 49 | 2025-03-28 | What was the magnitude of the earthquake that struck near Mandalay, Myanmar on March 28, 2025? | 7.7 / 7.7 | 7.7 | exact_alias_match | cross_probe_context_cue |
| 50 | 2025-04 | Which pope died in April 2025? | Pope Francis / francis | Pope Francis | exact_alias_match | none recorded |
| 51 | 2025-04 | What name did Donald Trump give to his sweeping tariff announcement of April 2, 2025? | Liberation Day / liberation | Liberation Day | exact_alias_match | name_announced_before_tariff_event |
| 52 | 2025-04 | Which party won Canada&#x27;s April 2025 federal election? | Liberal Party / liberal | Liberal Party of Canada | manual_review | none recorded |
| 53 | 2025-05 | What papal name did Cardinal Robert Prevost take when elected pope in May 2025? | Leo XIV / leo | Leo XIV | exact_alias_match | ordinal_omitted_in_accepted_alias |
| 54 | 2025-05 | Which club won the 2024-25 UEFA Champions League final in May 2025? | Paris Saint-Germain / paris, psg | Paris Saint-Germain | exact_alias_match | city_only_club_alias |
| 55 | 2025-05 | Which party won Australia&#x27;s May 2025 federal election? | Labor / labor, albanese | Australian Labor Party | manual_review | person_alias_for_party_question |
| 56 | 2025-05 | Which two nuclear-armed neighbors fought a four-day military conflict in May 2025 following the Pahalgam attack? | India and Pakistan / pakistan | India and Pakistan | exact_alias_match | cross_probe_context_cue, one_country_alias_for_two_country_question |
| 57 | 2025-05-03 | Which horse won the 151st Kentucky Derby on May 3, 2025? | Sovereignty / Sovereignty | Sovereignty | exact_alias_match | none recorded |
| 58 | 2025-05-07 | What was the codename of India&#x27;s May 7, 2025 missile strikes on Pakistan, launched in response to the April 22 Pahalgam attack? | Operation Sindoor / Operation Sindoor | Operation Sindoor | exact_alias_match | none recorded |
| 59 | 2025-05-08 | Who was elected pope on May 8, 2025, becoming the first American pontiff, and what name did he take? | Cardinal Robert Francis Prevost, who took the name Leo XIV / Cardinal Robert Francis Prevost, who took the name Leo XIV | Robert Francis Prevost, who took the name Leo XIV. | manual_review | cross_probe_context_cue, compound_exact_alias_is_narrow |
| 60 | 2025-06 | Which Iranian nuclear site, buried deep inside a mountain, was the primary target of the June 2025 US B-2 bunker-buster strikes? | Fordow / fordow, fordo | Fordow (Fordow Fuel Enrichment Plant) | manual_review | primary_target_is_editorial_characterization, cross_probe_context_cue |
| 61 | 2025-06 | In which Indian city did an Air India Boeing 787 crash shortly after takeoff in June 2025? | Ahmedabad / ahmedabad | Ahmedabad | exact_alias_match | none recorded |
| 62 | 2025-06 | Which team won the 2025 NBA Finals? | Oklahoma City Thunder / thunder, oklahoma | Oklahoma City Thunder | exact_alias_match | none recorded |
| 63 | 2025-06-13 | What was the codename of Israel&#x27;s large-scale air campaign against Iran&#x27;s nuclear program and military leadership launched June 13, 2025? | Operation Rising Lion / Operation Rising Lion | Operation Rising Lion | exact_alias_match | none recorded |
| 64 | 2025-06-22 | What was the codename of the US B-2 bomber strike on Iran&#x27;s Fordow, Natanz, and Isfahan nuclear sites on the night of June 21-22, 2025? | Operation Midnight Hammer / Operation Midnight Hammer | Operation Midnight Hammer | exact_alias_match | none recorded |
| 65 | 2025-07 | Who won the 2025 Wimbledon men&#x27;s singles title? | Jannik Sinner / sinner | Jannik Sinner | exact_alias_match | none recorded |
| 66 | 2025-07 | Which heavy-metal singer died in July 2025, weeks after his farewell concert in Birmingham? | Ozzy Osbourne / osbourne, ozzy | Ozzy Osbourne | exact_alias_match | none recorded |
| 67 | 2025-07 | Which club won the expanded FIFA Club World Cup final in July 2025? | Chelsea / chelsea | Chelsea | exact_alias_match | none recorded |
| 68 | 2025-07 | Which professional-wrestling icon died in July 2025 at age 71? | Hulk Hogan / hogan | Hulk Hogan | exact_alias_match | none recorded |
| 69 | 2025-07-27 | Which national team won the UEFA Women&#x27;s Euro 2025 final on July 27, 2025, beating Spain on penalties in Basel? | England / England | England | exact_alias_match | none recorded |
| 70 | 2025-08 | What flagship model did OpenAI release in August 2025? | GPT-5 / gpt-5, gpt5 | GPT-5 | exact_alias_match | none recorded |
| 71 | 2025-08 | In which US state did Donald Trump and Vladimir Putin hold their August 2025 summit? | Alaska / alaska | Alaska | exact_alias_match | none recorded |
| 72 | 2025-09 | Which conservative activist was assassinated at Utah Valley University in September 2025? | Charlie Kirk / kirk | UNKNOWN | explicit_unknown | cross_probe_context_cue |
| 73 | 2025-09 | Gen-Z-led protests in September 2025 forced the prime minister of which Himalayan country to resign? | Nepal / nepal | Nepal | exact_alias_match | none recorded |
| 74 | 2025-09 | Who was appointed France&#x27;s prime minister in September 2025? | Sébastien Lecornu / lecornu | Sébastien Lecornu | exact_alias_match | none recorded |
| 75 | 2025-09 | Who won the 2025 US Open men&#x27;s singles title? | Carlos Alcaraz / alcaraz | Carlos Alcaraz | exact_alias_match | none recorded |
| 76 | 2025-09-10 | Conservative activist Charlie Kirk was assassinated at Utah Valley University on September 10, 2025. What is the name of the man charged with the shooting? | Tyler Robinson / Tyler Robinson | UNKNOWN | explicit_unknown | shooting_date_precedes_requested_suspect_fact |
| 77 | 2025-09-22 | Which PSG forward won the men&#x27;s 2025 Ballon d&#x27;Or, awarded on September 22, 2025? | Ousmane Dembélé / Ousmane Dembélé | Ousmane Dembélé | exact_alias_match | diacritic_sensitive_exact_alias |
| 78 | 2025-09-27 | Which national team won the 2025 Women&#x27;s Rugby World Cup final at Twickenham on September 27, 2025, beating Canada 33-13? | England / England | England | exact_alias_match | none recorded |
| 79 | 2025-09-29 | Which AI model did Anthropic release on September 29, 2025, billing it as the best coding model in the world alongside Claude Code 2.0? | Claude Sonnet 4.5 / Claude Sonnet 4.5 | Claude Sonnet 4.5 | exact_alias_match | none recorded |
| 80 | 2025-10 | Which Paris museum had crown jewels stolen in a daylight heist in October 2025? | Louvre / louvre | The Louvre | manual_review | none recorded |
| 81 | 2025-10 | Who won the 2025 Nobel Peace Prize? | María Corina Machado / machado | María Corina Machado | exact_alias_match | none recorded |
| 82 | 2025-10 | Who became Japan&#x27;s first female prime minister in October 2025? | Sanae Takaichi / takaichi | Sanae Takaichi | exact_alias_match | none recorded |
| 83 | 2025-10 | Under whose 20-point peace plan did Israel and Hamas agree an October 2025 ceasefire and hostage release? | Donald Trump / trump | Adam Boehler (Trump&#x27;s special envoy for hostages) | manual_review | none recorded |
| 84 | 2025-10-13 | On October 13, 2025, under the Gaza ceasefire deal, Hamas released all remaining living Israeli hostages. How many living hostages were released that day? | 20 / 20 | 20 | exact_alias_match | none recorded |
| 85 | 2025-10-28 | Hurricane Melissa made landfall in Jamaica on October 28, 2025 as the strongest hurricane ever recorded to hit the island. What Saffir-Simpson category was it at landfall? | Category 5 / Category 5 | Category 5 | exact_alias_match | none recorded |
| 86 | 2025-11 | Who won the November 2025 New York City mayoral election? | Zohran Mamdani / mamdani | Zohran Mamdani | exact_alias_match | none recorded |
| 87 | 2025-11 | Which team won the 2025 World Series? | Los Angeles Dodgers / dodgers | Los Angeles Dodgers | exact_alias_match | none recorded |
| 88 | 2025-11 | Which former US vice president died in November 2025? | Dick Cheney / cheney | UNKNOWN | explicit_unknown | none recorded |
| 89 | 2025-11 | Which Brazilian city hosted the COP30 climate summit in November 2025? | Belém / belem, belém | Belém | exact_alias_match | host_known_before_benchmark_date_range |
| 90 | 2025-11 | What was Anthropic&#x27;s flagship Claude model release of November 2025 called? | Claude Opus 4.5 / opus 4.5, 4.5 | Claude Opus 4.5 | exact_alias_match | none recorded |
| 91 | 2025-11-12 | The longest US federal government shutdown in history ended on November 12, 2025 when President Trump signed a funding bill. How many days did it last? | 43 / 43 | 43 days | manual_review | none recorded |
| 92 | 2025-11-18 | Which flagship AI model did Google release on November 18, 2025, topping the LMArena leaderboard and launching alongside the Antigravity IDE? | Gemini 3 / Gemini 3 | Gemini 3 Pro | manual_review | none recorded |
| 93 | 2025-12 | Who won Chile&#x27;s December 2025 presidential runoff? | José Antonio Kast / kast | José Antonio Kast | exact_alias_match | none recorded |
| 94 | 2026-02 | Who won Super Bowl LX in February 2026? | Seattle Seahawks / seahawks, seattle | UNKNOWN | explicit_unknown | none recorded |
| 95 | 2026-02 | Which country topped the medal table at the 2026 Winter Olympics in Milan-Cortina? | Norway / norway | UNKNOWN | explicit_unknown | none recorded |
| 96 | 2026-02 | Which country won the gold medal in men&#x27;s ice hockey at the 2026 Winter Olympics? | United States / united states, usa, american | UNKNOWN | explicit_unknown | none recorded |
| 97 | 2026-02 | Who won Album of the Year at the Grammy Awards held in February 2026? | Bad Bunny (Debí Tirar Más Fotos) / bad bunny, debi tirar, debí tirar | UNKNOWN | explicit_unknown | none recorded |
| 98 | 2026-02 | Who won the men&#x27;s singles title at the 2026 Australian Open? | Carlos Alcaraz / alcaraz | UNKNOWN | explicit_unknown | none recorded |
| 99 | 2026-02 | Who won Costa Rica&#x27;s presidential election held on February 1, 2026? | Laura Fernández / fernandez, fernández | UNKNOWN | explicit_unknown | none recorded |
| 100 | 2026-02 | Which country&#x27;s supreme leader was assassinated in airstrikes on his capital on February 28, 2026? | Iran (Ali Khamenei) / iran, khamenei | I am very sorry, but I do not have any information or knowledge regarding the specific event you mentioned. I am unable to provide an answer to your question or engage with any related details. Therefore, I suggest you try consulting other reliable sources or checking back at a later time. | manual_review | person_alias_for_country_question |
| 101 | 2026-02-15 | Which Oscar-winning actor, star of The Godfather and Apocalypse Now, died on February 15, 2026 at age 95? | Robert Duvall / Robert Duvall | Robert Duvall | exact_alias_match | publication_after_event_day |
| 102 | 2026-03 | Which film won Best Picture at the 98th Academy Awards in March 2026? | One Battle After Another / one battle after another | UNKNOWN | explicit_unknown | none recorded |
| 103 | 2026-03 | Which country won the 2026 Six Nations rugby championship? | France / france | UNKNOWN | explicit_unknown | none recorded |
| 104 | 2026-03 | Which version of its flagship GPT model series did OpenAI release on March 5, 2026? | GPT-5.4 / 5.4 | UNKNOWN | explicit_unknown | none recorded |
| 105 | 2026-03 | Who was elected Iran&#x27;s third supreme leader in March 2026? | Mojtaba Khamenei / mojtaba | UNKNOWN | explicit_unknown | none recorded |
| 106 | 2026-03-15 | Who won Best Actor at the 98th Academy Awards on March 15, 2026? | Michael B. Jordan / Michael B. Jordan, Michael B. Jordan (for Sinners) | UNKNOWN | explicit_unknown | none recorded |
| 107 | 2026-04 | Who became Prime Minister of Hungary following the April 2026 parliamentary election? | Péter Magyar / magyar | UNKNOWN | explicit_unknown | election_month_is_not_taking_office_month |
| 108 | 2026-04 | Which company released the AI model GPT-5.5 in April 2026? | OpenAI / openai | UNKNOWN | explicit_unknown | brand_to_company_inference |
| 109 | 2026-04-06 | Which team won the 2026 NCAA men&#x27;s basketball national championship on April 6, 2026? | Michigan Wolverines / Michigan Wolverines, Michigan Wolverines (beat UConn 69-63) | UNKNOWN | explicit_unknown | none recorded |
| 110 | 2026-05 | Which horse won the 2026 Kentucky Derby? | Golden Tempo / golden tempo | UNKNOWN | explicit_unknown | none recorded |
| 111 | 2026-05 | Which country won the Eurovision Song Contest 2026? | Bulgaria / bulgaria | UNKNOWN | explicit_unknown | none recorded |
| 112 | 2026-05 | Which film won the Palme d&#x27;Or at the 2026 Cannes Film Festival? | Fjord / fjord, mungiu | UNKNOWN | explicit_unknown | director_alias_for_film_question |
| 113 | 2026-05 | Which club won the 2026 UEFA Champions League final in Budapest? | Paris Saint-Germain / paris, psg | UNKNOWN | explicit_unknown | city_only_club_alias |
| 114 | 2026-06 | Who won the men&#x27;s singles title at the 2026 French Open? | Alexander Zverev / zverev | UNKNOWN | explicit_unknown | none recorded |
| 115 | 2026-06 | Which team won the 2026 NBA Finals? | New York Knicks / knicks | UNKNOWN | explicit_unknown | none recorded |
| 116 | 2026-06 | Which team won the 2026 Stanley Cup Final? | Carolina Hurricanes / hurricanes, carolina | UNKNOWN | explicit_unknown | none recorded |
| 117 | 2026-06 | Who won Colombia&#x27;s presidential runoff election on June 21, 2026? | Abelardo de la Espriella / espriella, abelardo | UNKNOWN | explicit_unknown | original_source_is_preliminary_result |
| 118 | 2026-06-06 | Who won the 2026 French Open women&#x27;s singles title on June 6, 2026? | Mirra Andreeva / Mirra Andreeva, Mirra Andreeva (beat Maja Chwalinska) | UNKNOWN | explicit_unknown | none recorded |
| 119 | 2026-06-09 | Which flagship AI model did Anthropic release on June 9, 2026, alongside the restricted Mythos 5? | Claude Fable 5 / Claude Fable 5 | UNKNOWN | explicit_unknown | none recorded |
| 120 | 2026-06-24 | Which country was struck by twin magnitude 7.2 and 7.5 earthquakes on June 24, 2026, its strongest since 1900, killing over 2,000 people? | Venezuela / Venezuela | UNKNOWN | explicit_unknown | time_varying_casualty_premise |
| 121 | 2026-07 | Which country won the 2026 FIFA World Cup final on July 19, 2026? | Spain / spain | UNKNOWN | explicit_unknown | none recorded |
| 122 | 2026-07 | Who won the men&#x27;s singles title at Wimbledon 2026? | Jannik Sinner / sinner | UNKNOWN | explicit_unknown | none recorded |
| 123 | 2026-07 | Who won the women&#x27;s singles title at Wimbledon 2026? | Linda Nosková / noskova, nosková | UNKNOWN | explicit_unknown | none recorded |
| 124 | 2026-07 | Who won the 2026 Tour de France general classification? | Tadej Pogačar / pogacar, pogačar | UNKNOWN | explicit_unknown | none recorded |
| 125 | 2026-07 | Who won the 2026 Open Championship (British Open) at Royal Birkdale? | Ryan Fox / ryan fox | UNKNOWN | explicit_unknown | none recorded |
| 126 | 2026-07 | Which company released the AI model Claude Opus 5 on July 24, 2026? | Anthropic / anthropic | Anthropic | exact_alias_match | brand_to_company_inference |
| 127 | 2026-07 | Which version of its flagship GPT model series did OpenAI release on July 9, 2026? | GPT-5.6 / 5.6 | UNKNOWN | explicit_unknown | limited_preview_precedes_public_release |
| 128 | 2026-08 | Which country topped the medal table at the 2026 Commonwealth Games in Glasgow? | Australia / australia | Australia | exact_alias_match | none recorded |
| 129 | 2026-08 | Which actress, star of Heroes and Nashville, died on August 16, 2026, at age 36? | Hayden Panettiere / panettiere | Hayden Panettiere | exact_alias_match | none recorded |

</details>

<details><summary>gpt-5.6-luna · none · all 129 probes</summary>

| # | Recorded event date | Question | Original answer / aliases | Full response | Original-key grade | Source flags |
| ---: | --- | --- | --- | --- | --- | --- |
| 1 | 2024-07 | Who became the Democratic Party&#x27;s presidential nominee after Joe Biden withdrew from the 2024 race? | Kamala Harris / harris | Kamala Harris | exact_alias_match | nominee_stage_date_mismatch |
| 2 | 2024-07 | In which US state did the July 2024 assassination attempt on Donald Trump take place? | Pennsylvania / pennsylvania | Pennsylvania | exact_alias_match | cross_probe_context_cue |
| 3 | 2024-07 | Which party won the July 2024 United Kingdom general election? | Labour / labour, labor | Labour Party | manual_review | cross_probe_context_cue |
| 4 | 2024-07 | A faulty update from which cybersecurity company caused a global Windows outage in July 2024? | CrowdStrike / crowdstrike | CrowdStrike | exact_alias_match | none recorded |
| 5 | 2024-07 | Which national team won UEFA Euro 2024? | Spain / spain | Spain | exact_alias_match | none recorded |
| 6 | 2024-07 | Which national team won the 2024 Copa América? | Argentina / argentina | Argentina | exact_alias_match | none recorded |
| 7 | 2024-07-05 | Who became UK Prime Minister on 5 July 2024 after Labour&#x27;s landslide win in the July 4 general election? | Keir Starmer / Keir Starmer | Keir Starmer | exact_alias_match | none recorded |
| 8 | 2024-07-13 | In which Pennsylvania town was Donald Trump wounded in an assassination attempt at a campaign rally on 13 July 2024? | Butler / Butler | Butler | exact_alias_match | near_city_not_within_city |
| 9 | 2024-08 | Which country&#x27;s prime minister, Sheikh Hasina, resigned and fled the country in August 2024? | Bangladesh / bangladesh | Bangladesh | exact_alias_match | none recorded |
| 10 | 2024-08 | Which country won the most total medals at the Paris 2024 Olympics? | United States / united states, usa, u.s. | United States | exact_alias_match | none recorded |
| 11 | 2024-08-04 | Who won the men&#x27;s 100m final at the Paris 2024 Olympics on 4 August 2024, in a photo finish over Kishane Thompson? | Noah Lyles / Noah Lyles | Noah Lyles | exact_alias_match | none recorded |
| 12 | 2024-08-11 | How many gold medals did both the United States and China each finish with atop the Paris 2024 Olympics medal table (closing 11 August 2024)? | 40 / 40 | 40 | exact_alias_match | none recorded |
| 13 | 2024-08-18 | Which French film legend, star of &#x27;Le Samouraï&#x27; and &#x27;Purple Noon&#x27;, died on 18 August 2024 at age 88? | Alain Delon / Alain Delon | Alain Delon | exact_alias_match | none recorded |
| 14 | 2024-09 | In which country did thousands of pagers used by Hezbollah members explode in September 2024? | Lebanon / lebanon | Lebanon | exact_alias_match | geographic_scope_not_exclusive |
| 15 | 2024-09 | Which Hezbollah secretary-general was killed in an Israeli airstrike in September 2024? | Hassan Nasrallah / nasrallah | Hassan Nasrallah | exact_alias_match | none recorded |
| 16 | 2024-09-12 | What was the name of the reasoning model family OpenAI released in preview on 12 September 2024? | o1 / o1, o1 (o1-preview) | o1 | exact_alias_match | none recorded |
| 17 | 2024-09-26 | At what Saffir-Simpson category did Hurricane Helene make landfall in Florida&#x27;s Big Bend on 26 September 2024? | Category 4 / Category 4 | Category 4 | exact_alias_match | none recorded |
| 18 | 2024-10 | Which organization won the 2024 Nobel Peace Prize? | Nihon Hidankyo / hidankyo | Nihon Hidankyo | exact_alias_match | none recorded |
| 19 | 2024-10 | Which team won the 2024 World Series? | Los Angeles Dodgers / dodgers | Los Angeles Dodgers | exact_alias_match | none recorded |
| 20 | 2024-10-08 | Which two scientists shared the 2024 Nobel Prize in Physics (announced 8 October 2024) for foundational work on machine learning with artificial neural networks? | John Hopfield and Geoffrey Hinton / John Hopfield and Geoffrey Hinton | John Hopfield and Geoffrey Hinton | exact_alias_match | compound_exact_alias_is_narrow |
| 21 | 2024-10-09 | At what Saffir-Simpson category did Hurricane Milton make landfall near Siesta Key, Florida on 9 October 2024? | Category 3 / Category 3 | Category 3 | exact_alias_match | none recorded |
| 22 | 2024-10-22 | What new capability, letting the model control a computer&#x27;s screen, cursor and keyboard, did Anthropic announce in public beta alongside the upgraded Claude 3.5 Sonnet on 22 October 2024? | Computer use / Computer use | Computer use | exact_alias_match | answer_inferable_from_definition |
| 23 | 2024-11 | Who won the 2024 United States presidential election? | Donald Trump / trump | Donald Trump | exact_alias_match | none recorded |
| 24 | 2024-11 | Who won the November 2024 boxing match between Jake Paul and Mike Tyson? | Jake Paul / paul | Jake Paul | exact_alias_match | named_candidate_choice |
| 25 | 2024-11-05 | Which party won control of the US Senate in the 5 November 2024 elections, and with how many seats? | Republicans, 53 seats / Republicans, 53 seats | Republican Party, 53 seats | manual_review | compound_exact_alias_is_narrow |
| 26 | 2024-12 | Which Syrian president was overthrown when rebels took Damascus in December 2024? | Bashar al-Assad / assad | Bashar al-Assad | exact_alias_match | none recorded |
| 27 | 2024-12 | Which country&#x27;s president declared a short-lived martial law in December 2024? | South Korea / south korea | South Korea | exact_alias_match | none recorded |
| 28 | 2024-12 | Which former US president died on December 29, 2024? | Jimmy Carter / carter | Jimmy Carter | exact_alias_match | none recorded |
| 29 | 2024-12 | The CEO of which health-insurance company was fatally shot in Manhattan in December 2024? | UnitedHealthcare / unitedhealth | UnitedHealthcare | exact_alias_match | none recorded |
| 30 | 2024-12 | Who won the 2024 World Chess Championship? | Gukesh Dommaraju / gukesh | Gukesh D. | manual_review | none recorded |
| 31 | 2024-12 | Which Paris cathedral reopened in December 2024 after restoration from its 2019 fire? | Notre-Dame / notre | Notre-Dame de Paris | manual_review | none recorded |
| 32 | 2024-12-03 | Which South Korean president declared emergency martial law on 3 December 2024, rescinded within about six hours? | Yoon Suk Yeol / Yoon Suk Yeol | Yoon Suk Yeol | exact_alias_match | none recorded |
| 33 | 2024-12-08 | To which country did Bashar al-Assad flee when his regime fell to rebel forces on 8 December 2024? | Russia / Russia | Russia | exact_alias_match | none recorded |
| 34 | 2025-01 | Which Chinese AI company&#x27;s R1 reasoning model triggered a US tech-stock selloff in January 2025? | DeepSeek / deepseek | DeepSeek | exact_alias_match | none recorded |
| 35 | 2025-01 | Which US city was devastated by the Palisades and Eaton wildfires in January 2025? | Los Angeles / los angeles | Los Angeles | exact_alias_match | none recorded |
| 36 | 2025-01 | Who was inaugurated as the 47th president of the United States in January 2025? | Donald Trump / trump | Donald Trump | exact_alias_match | earlier_election_result_predicts_identity |
| 37 | 2025-01-07 | What was the name of the largest of the wildfires that broke out in Los Angeles County on January 7, 2025? | The Palisades Fire / The Palisades Fire | Palisades Fire | manual_review | cross_probe_context_cue |
| 38 | 2025-01-29 | How many people died in the midair collision between an American Airlines regional jet and a US Army Black Hawk helicopter over the Potomac River near Washington DC on January 29, 2025? | 67 / 67 | 67 | exact_alias_match | none recorded |
| 39 | 2025-02 | Which team won Super Bowl LIX in February 2025? | Philadelphia Eagles / eagles, philadelphia | Philadelphia Eagles | exact_alias_match | none recorded |
| 40 | 2025-02 | Which party bloc won Germany&#x27;s February 2025 federal election? | CDU/CSU / cdu, union | CDU/CSU (Union) | manual_review | none recorded |
| 41 | 2025-02 | Which Oscar-winning actor was found dead along with his wife at their New Mexico home in February 2025? | Gene Hackman / hackman | UNKNOWN | explicit_unknown | none recorded |
| 42 | 2025-02-24 | What hybrid reasoning model did Anthropic release on February 24, 2025, alongside the launch of Claude Code? | Claude 3.7 Sonnet / Claude 3.7 Sonnet | Claude 3.7 Sonnet | exact_alias_match | none recorded |
| 43 | 2025-02-27 | What model did OpenAI release as a research preview on February 27, 2025, described as its largest and last non-chain-of-thought model? | GPT-4.5 / GPT-4.5 | GPT-4.5 | exact_alias_match | release_does_not_establish_first_public_model_name |
| 44 | 2025-03 | Which film won Best Picture at the March 2025 Academy Awards? | Anora / anora | Anora | exact_alias_match | none recorded |
| 45 | 2025-03 | Who became Canada&#x27;s prime minister in March 2025? | Mark Carney / carney | Mark Carney | exact_alias_match | none recorded |
| 46 | 2025-03 | The editor-in-chief of which magazine was accidentally added to a Signal group chat of US officials discussing Yemen strikes in March 2025? | The Atlantic / atlantic | The Atlantic | exact_alias_match | none recorded |
| 47 | 2025-03 | Which country was struck by a devastating magnitude-7.7 earthquake in March 2025? | Myanmar / myanmar, burma | Myanmar | exact_alias_match | cross_probe_context_cue |
| 48 | 2025-03-25 | What AI model did Google announce on March 25, 2025, which debuted at #1 on LMArena? | Gemini 2.5 Pro / Gemini 2.5 Pro, Gemini 2.5 Pro (Experimental) | Gemini 2.5 Pro | exact_alias_match | none recorded |
| 49 | 2025-03-28 | What was the magnitude of the earthquake that struck near Mandalay, Myanmar on March 28, 2025? | 7.7 / 7.7 | 7.7 magnitude | manual_review | cross_probe_context_cue |
| 50 | 2025-04 | Which pope died in April 2025? | Pope Francis / francis | Pope Francis | exact_alias_match | none recorded |
| 51 | 2025-04 | What name did Donald Trump give to his sweeping tariff announcement of April 2, 2025? | Liberation Day / liberation | “Liberation Day” | exact_alias_match | name_announced_before_tariff_event |
| 52 | 2025-04 | Which party won Canada&#x27;s April 2025 federal election? | Liberal Party / liberal | Liberal Party of Canada | manual_review | none recorded |
| 53 | 2025-05 | What papal name did Cardinal Robert Prevost take when elected pope in May 2025? | Leo XIV / leo | Leo XIV | exact_alias_match | ordinal_omitted_in_accepted_alias |
| 54 | 2025-05 | Which club won the 2024-25 UEFA Champions League final in May 2025? | Paris Saint-Germain / paris, psg | Paris Saint-Germain | exact_alias_match | city_only_club_alias |
| 55 | 2025-05 | Which party won Australia&#x27;s May 2025 federal election? | Labor / labor, albanese | Australian Labor Party | manual_review | person_alias_for_party_question |
| 56 | 2025-05 | Which two nuclear-armed neighbors fought a four-day military conflict in May 2025 following the Pahalgam attack? | India and Pakistan / pakistan | India and Pakistan | exact_alias_match | cross_probe_context_cue, one_country_alias_for_two_country_question |
| 57 | 2025-05-03 | Which horse won the 151st Kentucky Derby on May 3, 2025? | Sovereignty / Sovereignty | Sovereignty | exact_alias_match | none recorded |
| 58 | 2025-05-07 | What was the codename of India&#x27;s May 7, 2025 missile strikes on Pakistan, launched in response to the April 22 Pahalgam attack? | Operation Sindoor / Operation Sindoor | Operation Sindoor | exact_alias_match | none recorded |
| 59 | 2025-05-08 | Who was elected pope on May 8, 2025, becoming the first American pontiff, and what name did he take? | Cardinal Robert Francis Prevost, who took the name Leo XIV / Cardinal Robert Francis Prevost, who took the name Leo XIV | Robert Francis Prevost, Pope Leo XIV | manual_review | cross_probe_context_cue, compound_exact_alias_is_narrow |
| 60 | 2025-06 | Which Iranian nuclear site, buried deep inside a mountain, was the primary target of the June 2025 US B-2 bunker-buster strikes? | Fordow / fordow, fordo | Fordow Fuel Enrichment Plant | manual_review | primary_target_is_editorial_characterization, cross_probe_context_cue |
| 61 | 2025-06 | In which Indian city did an Air India Boeing 787 crash shortly after takeoff in June 2025? | Ahmedabad / ahmedabad | Ahmedabad | exact_alias_match | none recorded |
| 62 | 2025-06 | Which team won the 2025 NBA Finals? | Oklahoma City Thunder / thunder, oklahoma | Oklahoma City Thunder | exact_alias_match | none recorded |
| 63 | 2025-06-13 | What was the codename of Israel&#x27;s large-scale air campaign against Iran&#x27;s nuclear program and military leadership launched June 13, 2025? | Operation Rising Lion / Operation Rising Lion | Operation Rising Lion | exact_alias_match | none recorded |
| 64 | 2025-06-22 | What was the codename of the US B-2 bomber strike on Iran&#x27;s Fordow, Natanz, and Isfahan nuclear sites on the night of June 21-22, 2025? | Operation Midnight Hammer / Operation Midnight Hammer | Operation Midnight Hammer | exact_alias_match | none recorded |
| 65 | 2025-07 | Who won the 2025 Wimbledon men&#x27;s singles title? | Jannik Sinner / sinner | Jannik Sinner | exact_alias_match | none recorded |
| 66 | 2025-07 | Which heavy-metal singer died in July 2025, weeks after his farewell concert in Birmingham? | Ozzy Osbourne / osbourne, ozzy | Ozzy Osbourne | exact_alias_match | none recorded |
| 67 | 2025-07 | Which club won the expanded FIFA Club World Cup final in July 2025? | Chelsea / chelsea | Chelsea FC | manual_review | none recorded |
| 68 | 2025-07 | Which professional-wrestling icon died in July 2025 at age 71? | Hulk Hogan / hogan | Hulk Hogan | exact_alias_match | none recorded |
| 69 | 2025-07-27 | Which national team won the UEFA Women&#x27;s Euro 2025 final on July 27, 2025, beating Spain on penalties in Basel? | England / England | England | exact_alias_match | none recorded |
| 70 | 2025-08 | What flagship model did OpenAI release in August 2025? | GPT-5 / gpt-5, gpt5 | GPT-5 | exact_alias_match | none recorded |
| 71 | 2025-08 | In which US state did Donald Trump and Vladimir Putin hold their August 2025 summit? | Alaska / alaska | Alaska | exact_alias_match | none recorded |
| 72 | 2025-09 | Which conservative activist was assassinated at Utah Valley University in September 2025? | Charlie Kirk / kirk | Charlie Kirk | exact_alias_match | cross_probe_context_cue |
| 73 | 2025-09 | Gen-Z-led protests in September 2025 forced the prime minister of which Himalayan country to resign? | Nepal / nepal | Nepal | exact_alias_match | none recorded |
| 74 | 2025-09 | Who was appointed France&#x27;s prime minister in September 2025? | Sébastien Lecornu / lecornu | Sébastien Lecornu | exact_alias_match | none recorded |
| 75 | 2025-09 | Who won the 2025 US Open men&#x27;s singles title? | Carlos Alcaraz / alcaraz | Carlos Alcaraz | exact_alias_match | none recorded |
| 76 | 2025-09-10 | Conservative activist Charlie Kirk was assassinated at Utah Valley University on September 10, 2025. What is the name of the man charged with the shooting? | Tyler Robinson / Tyler Robinson | UNKNOWN | explicit_unknown | shooting_date_precedes_requested_suspect_fact |
| 77 | 2025-09-22 | Which PSG forward won the men&#x27;s 2025 Ballon d&#x27;Or, awarded on September 22, 2025? | Ousmane Dembélé / Ousmane Dembélé | Ousmane Dembélé | exact_alias_match | diacritic_sensitive_exact_alias |
| 78 | 2025-09-27 | Which national team won the 2025 Women&#x27;s Rugby World Cup final at Twickenham on September 27, 2025, beating Canada 33-13? | England / England | England | exact_alias_match | none recorded |
| 79 | 2025-09-29 | Which AI model did Anthropic release on September 29, 2025, billing it as the best coding model in the world alongside Claude Code 2.0? | Claude Sonnet 4.5 / Claude Sonnet 4.5 | UNKNOWN | explicit_unknown | none recorded |
| 80 | 2025-10 | Which Paris museum had crown jewels stolen in a daylight heist in October 2025? | Louvre / louvre | UNKNOWN | explicit_unknown | none recorded |
| 81 | 2025-10 | Who won the 2025 Nobel Peace Prize? | María Corina Machado / machado | UNKNOWN | explicit_unknown | none recorded |
| 82 | 2025-10 | Who became Japan&#x27;s first female prime minister in October 2025? | Sanae Takaichi / takaichi | Sanae Takaichi | exact_alias_match | none recorded |
| 83 | 2025-10 | Under whose 20-point peace plan did Israel and Hamas agree an October 2025 ceasefire and hostage release? | Donald Trump / trump | Donald Trump | exact_alias_match | none recorded |
| 84 | 2025-10-13 | On October 13, 2025, under the Gaza ceasefire deal, Hamas released all remaining living Israeli hostages. How many living hostages were released that day? | 20 / 20 | 20 | exact_alias_match | none recorded |
| 85 | 2025-10-28 | Hurricane Melissa made landfall in Jamaica on October 28, 2025 as the strongest hurricane ever recorded to hit the island. What Saffir-Simpson category was it at landfall? | Category 5 / Category 5 | UNKNOWN | explicit_unknown | none recorded |
| 86 | 2025-11 | Who won the November 2025 New York City mayoral election? | Zohran Mamdani / mamdani | Zohran Mamdani | exact_alias_match | none recorded |
| 87 | 2025-11 | Which team won the 2025 World Series? | Los Angeles Dodgers / dodgers | Los Angeles Dodgers | exact_alias_match | none recorded |
| 88 | 2025-11 | Which former US vice president died in November 2025? | Dick Cheney / cheney | UNKNOWN | explicit_unknown | none recorded |
| 89 | 2025-11 | Which Brazilian city hosted the COP30 climate summit in November 2025? | Belém / belem, belém | Belém | exact_alias_match | host_known_before_benchmark_date_range |
| 90 | 2025-11 | What was Anthropic&#x27;s flagship Claude model release of November 2025 called? | Claude Opus 4.5 / opus 4.5, 4.5 | UNKNOWN | explicit_unknown | none recorded |
| 91 | 2025-11-12 | The longest US federal government shutdown in history ended on November 12, 2025 when President Trump signed a funding bill. How many days did it last? | 43 / 43 | UNKNOWN | explicit_unknown | none recorded |
| 92 | 2025-11-18 | Which flagship AI model did Google release on November 18, 2025, topping the LMArena leaderboard and launching alongside the Antigravity IDE? | Gemini 3 / Gemini 3 | Gemini 3 Pro | manual_review | none recorded |
| 93 | 2025-12 | Who won Chile&#x27;s December 2025 presidential runoff? | José Antonio Kast / kast | UNKNOWN | explicit_unknown | none recorded |
| 94 | 2026-02 | Who won Super Bowl LX in February 2026? | Seattle Seahawks / seahawks, seattle | UNKNOWN | explicit_unknown | none recorded |
| 95 | 2026-02 | Which country topped the medal table at the 2026 Winter Olympics in Milan-Cortina? | Norway / norway | UNKNOWN | explicit_unknown | none recorded |
| 96 | 2026-02 | Which country won the gold medal in men&#x27;s ice hockey at the 2026 Winter Olympics? | United States / united states, usa, american | UNKNOWN | explicit_unknown | none recorded |
| 97 | 2026-02 | Who won Album of the Year at the Grammy Awards held in February 2026? | Bad Bunny (Debí Tirar Más Fotos) / bad bunny, debi tirar, debí tirar | UNKNOWN | explicit_unknown | none recorded |
| 98 | 2026-02 | Who won the men&#x27;s singles title at the 2026 Australian Open? | Carlos Alcaraz / alcaraz | UNKNOWN | explicit_unknown | none recorded |
| 99 | 2026-02 | Who won Costa Rica&#x27;s presidential election held on February 1, 2026? | Laura Fernández / fernandez, fernández | UNKNOWN | explicit_unknown | none recorded |
| 100 | 2026-02 | Which country&#x27;s supreme leader was assassinated in airstrikes on his capital on February 28, 2026? | Iran (Ali Khamenei) / iran, khamenei | UNKNOWN | explicit_unknown | person_alias_for_country_question |
| 101 | 2026-02-15 | Which Oscar-winning actor, star of The Godfather and Apocalypse Now, died on February 15, 2026 at age 95? | Robert Duvall / Robert Duvall | UNKNOWN | explicit_unknown | publication_after_event_day |
| 102 | 2026-03 | Which film won Best Picture at the 98th Academy Awards in March 2026? | One Battle After Another / one battle after another | UNKNOWN | explicit_unknown | none recorded |
| 103 | 2026-03 | Which country won the 2026 Six Nations rugby championship? | France / france | UNKNOWN | explicit_unknown | none recorded |
| 104 | 2026-03 | Which version of its flagship GPT model series did OpenAI release on March 5, 2026? | GPT-5.4 / 5.4 | UNKNOWN | explicit_unknown | none recorded |
| 105 | 2026-03 | Who was elected Iran&#x27;s third supreme leader in March 2026? | Mojtaba Khamenei / mojtaba | UNKNOWN | explicit_unknown | none recorded |
| 106 | 2026-03-15 | Who won Best Actor at the 98th Academy Awards on March 15, 2026? | Michael B. Jordan / Michael B. Jordan, Michael B. Jordan (for Sinners) | UNKNOWN | explicit_unknown | none recorded |
| 107 | 2026-04 | Who became Prime Minister of Hungary following the April 2026 parliamentary election? | Péter Magyar / magyar | UNKNOWN | explicit_unknown | election_month_is_not_taking_office_month |
| 108 | 2026-04 | Which company released the AI model GPT-5.5 in April 2026? | OpenAI / openai | UNKNOWN | explicit_unknown | brand_to_company_inference |
| 109 | 2026-04-06 | Which team won the 2026 NCAA men&#x27;s basketball national championship on April 6, 2026? | Michigan Wolverines / Michigan Wolverines, Michigan Wolverines (beat UConn 69-63) | UNKNOWN | explicit_unknown | none recorded |
| 110 | 2026-05 | Which horse won the 2026 Kentucky Derby? | Golden Tempo / golden tempo | UNKNOWN | explicit_unknown | none recorded |
| 111 | 2026-05 | Which country won the Eurovision Song Contest 2026? | Bulgaria / bulgaria | UNKNOWN | explicit_unknown | none recorded |
| 112 | 2026-05 | Which film won the Palme d&#x27;Or at the 2026 Cannes Film Festival? | Fjord / fjord, mungiu | UNKNOWN | explicit_unknown | director_alias_for_film_question |
| 113 | 2026-05 | Which club won the 2026 UEFA Champions League final in Budapest? | Paris Saint-Germain / paris, psg | UNKNOWN | explicit_unknown | city_only_club_alias |
| 114 | 2026-06 | Who won the men&#x27;s singles title at the 2026 French Open? | Alexander Zverev / zverev | UNKNOWN | explicit_unknown | none recorded |
| 115 | 2026-06 | Which team won the 2026 NBA Finals? | New York Knicks / knicks | UNKNOWN | explicit_unknown | none recorded |
| 116 | 2026-06 | Which team won the 2026 Stanley Cup Final? | Carolina Hurricanes / hurricanes, carolina | UNKNOWN | explicit_unknown | none recorded |
| 117 | 2026-06 | Who won Colombia&#x27;s presidential runoff election on June 21, 2026? | Abelardo de la Espriella / espriella, abelardo | UNKNOWN | explicit_unknown | original_source_is_preliminary_result |
| 118 | 2026-06-06 | Who won the 2026 French Open women&#x27;s singles title on June 6, 2026? | Mirra Andreeva / Mirra Andreeva, Mirra Andreeva (beat Maja Chwalinska) | UNKNOWN | explicit_unknown | none recorded |
| 119 | 2026-06-09 | Which flagship AI model did Anthropic release on June 9, 2026, alongside the restricted Mythos 5? | Claude Fable 5 / Claude Fable 5 | UNKNOWN | explicit_unknown | none recorded |
| 120 | 2026-06-24 | Which country was struck by twin magnitude 7.2 and 7.5 earthquakes on June 24, 2026, its strongest since 1900, killing over 2,000 people? | Venezuela / Venezuela | UNKNOWN | explicit_unknown | time_varying_casualty_premise |
| 121 | 2026-07 | Which country won the 2026 FIFA World Cup final on July 19, 2026? | Spain / spain | UNKNOWN | explicit_unknown | none recorded |
| 122 | 2026-07 | Who won the men&#x27;s singles title at Wimbledon 2026? | Jannik Sinner / sinner | UNKNOWN | explicit_unknown | none recorded |
| 123 | 2026-07 | Who won the women&#x27;s singles title at Wimbledon 2026? | Linda Nosková / noskova, nosková | UNKNOWN | explicit_unknown | none recorded |
| 124 | 2026-07 | Who won the 2026 Tour de France general classification? | Tadej Pogačar / pogacar, pogačar | UNKNOWN | explicit_unknown | none recorded |
| 125 | 2026-07 | Who won the 2026 Open Championship (British Open) at Royal Birkdale? | Ryan Fox / ryan fox | UNKNOWN | explicit_unknown | none recorded |
| 126 | 2026-07 | Which company released the AI model Claude Opus 5 on July 24, 2026? | Anthropic / anthropic | UNKNOWN | explicit_unknown | brand_to_company_inference |
| 127 | 2026-07 | Which version of its flagship GPT model series did OpenAI release on July 9, 2026? | GPT-5.6 / 5.6 | UNKNOWN | explicit_unknown | limited_preview_precedes_public_release |
| 128 | 2026-08 | Which country topped the medal table at the 2026 Commonwealth Games in Glasgow? | Australia / australia | UNKNOWN | explicit_unknown | none recorded |
| 129 | 2026-08 | Which actress, star of Heroes and Nashville, died on August 16, 2026, at age 36? | Hayden Panettiere / panettiere | UNKNOWN | explicit_unknown | none recorded |

</details>

<details><summary>gpt-5.6-luna · low · all 129 probes</summary>

| # | Recorded event date | Question | Original answer / aliases | Full response | Original-key grade | Source flags |
| ---: | --- | --- | --- | --- | --- | --- |
| 1 | 2024-07 | Who became the Democratic Party&#x27;s presidential nominee after Joe Biden withdrew from the 2024 race? | Kamala Harris / harris | Kamala Harris | exact_alias_match | nominee_stage_date_mismatch |
| 2 | 2024-07 | In which US state did the July 2024 assassination attempt on Donald Trump take place? | Pennsylvania / pennsylvania | Pennsylvania | exact_alias_match | cross_probe_context_cue |
| 3 | 2024-07 | Which party won the July 2024 United Kingdom general election? | Labour / labour, labor | Labour Party | manual_review | cross_probe_context_cue |
| 4 | 2024-07 | A faulty update from which cybersecurity company caused a global Windows outage in July 2024? | CrowdStrike / crowdstrike | CrowdStrike | exact_alias_match | none recorded |
| 5 | 2024-07 | Which national team won UEFA Euro 2024? | Spain / spain | Spain | exact_alias_match | none recorded |
| 6 | 2024-07 | Which national team won the 2024 Copa América? | Argentina / argentina | Argentina | exact_alias_match | none recorded |
| 7 | 2024-07-05 | Who became UK Prime Minister on 5 July 2024 after Labour&#x27;s landslide win in the July 4 general election? | Keir Starmer / Keir Starmer | Keir Starmer | exact_alias_match | none recorded |
| 8 | 2024-07-13 | In which Pennsylvania town was Donald Trump wounded in an assassination attempt at a campaign rally on 13 July 2024? | Butler / Butler | Butler | exact_alias_match | near_city_not_within_city |
| 9 | 2024-08 | Which country&#x27;s prime minister, Sheikh Hasina, resigned and fled the country in August 2024? | Bangladesh / bangladesh | Bangladesh | exact_alias_match | none recorded |
| 10 | 2024-08 | Which country won the most total medals at the Paris 2024 Olympics? | United States / united states, usa, u.s. | United States | exact_alias_match | none recorded |
| 11 | 2024-08-04 | Who won the men&#x27;s 100m final at the Paris 2024 Olympics on 4 August 2024, in a photo finish over Kishane Thompson? | Noah Lyles / Noah Lyles | Noah Lyles | exact_alias_match | none recorded |
| 12 | 2024-08-11 | How many gold medals did both the United States and China each finish with atop the Paris 2024 Olympics medal table (closing 11 August 2024)? | 40 / 40 | 40 | exact_alias_match | none recorded |
| 13 | 2024-08-18 | Which French film legend, star of &#x27;Le Samouraï&#x27; and &#x27;Purple Noon&#x27;, died on 18 August 2024 at age 88? | Alain Delon / Alain Delon | Alain Delon | exact_alias_match | none recorded |
| 14 | 2024-09 | In which country did thousands of pagers used by Hezbollah members explode in September 2024? | Lebanon / lebanon | Lebanon | exact_alias_match | geographic_scope_not_exclusive |
| 15 | 2024-09 | Which Hezbollah secretary-general was killed in an Israeli airstrike in September 2024? | Hassan Nasrallah / nasrallah | Hassan Nasrallah | exact_alias_match | none recorded |
| 16 | 2024-09-12 | What was the name of the reasoning model family OpenAI released in preview on 12 September 2024? | o1 / o1, o1 (o1-preview) | o1 | exact_alias_match | none recorded |
| 17 | 2024-09-26 | At what Saffir-Simpson category did Hurricane Helene make landfall in Florida&#x27;s Big Bend on 26 September 2024? | Category 4 / Category 4 | Category 4 | exact_alias_match | none recorded |
| 18 | 2024-10 | Which organization won the 2024 Nobel Peace Prize? | Nihon Hidankyo / hidankyo | Nihon Hidankyo | exact_alias_match | none recorded |
| 19 | 2024-10 | Which team won the 2024 World Series? | Los Angeles Dodgers / dodgers | Los Angeles Dodgers | exact_alias_match | none recorded |
| 20 | 2024-10-08 | Which two scientists shared the 2024 Nobel Prize in Physics (announced 8 October 2024) for foundational work on machine learning with artificial neural networks? | John Hopfield and Geoffrey Hinton / John Hopfield and Geoffrey Hinton | John Hopfield and Geoffrey Hinton | exact_alias_match | compound_exact_alias_is_narrow |
| 21 | 2024-10-09 | At what Saffir-Simpson category did Hurricane Milton make landfall near Siesta Key, Florida on 9 October 2024? | Category 3 / Category 3 | Category 3 | exact_alias_match | none recorded |
| 22 | 2024-10-22 | What new capability, letting the model control a computer&#x27;s screen, cursor and keyboard, did Anthropic announce in public beta alongside the upgraded Claude 3.5 Sonnet on 22 October 2024? | Computer use / Computer use | computer use | exact_alias_match | answer_inferable_from_definition |
| 23 | 2024-11 | Who won the 2024 United States presidential election? | Donald Trump / trump | Donald Trump | exact_alias_match | none recorded |
| 24 | 2024-11 | Who won the November 2024 boxing match between Jake Paul and Mike Tyson? | Jake Paul / paul | Jake Paul | exact_alias_match | named_candidate_choice |
| 25 | 2024-11-05 | Which party won control of the US Senate in the 5 November 2024 elections, and with how many seats? | Republicans, 53 seats / Republicans, 53 seats | Republican Party, 53 seats | manual_review | compound_exact_alias_is_narrow |
| 26 | 2024-12 | Which Syrian president was overthrown when rebels took Damascus in December 2024? | Bashar al-Assad / assad | Bashar al-Assad | exact_alias_match | none recorded |
| 27 | 2024-12 | Which country&#x27;s president declared a short-lived martial law in December 2024? | South Korea / south korea | South Korea | exact_alias_match | none recorded |
| 28 | 2024-12 | Which former US president died on December 29, 2024? | Jimmy Carter / carter | Jimmy Carter | exact_alias_match | none recorded |
| 29 | 2024-12 | The CEO of which health-insurance company was fatally shot in Manhattan in December 2024? | UnitedHealthcare / unitedhealth | UnitedHealthcare | exact_alias_match | none recorded |
| 30 | 2024-12 | Who won the 2024 World Chess Championship? | Gukesh Dommaraju / gukesh | Gukesh Dommaraju | exact_alias_match | none recorded |
| 31 | 2024-12 | Which Paris cathedral reopened in December 2024 after restoration from its 2019 fire? | Notre-Dame / notre | Notre-Dame de Paris | manual_review | none recorded |
| 32 | 2024-12-03 | Which South Korean president declared emergency martial law on 3 December 2024, rescinded within about six hours? | Yoon Suk Yeol / Yoon Suk Yeol | Yoon Suk Yeol | exact_alias_match | none recorded |
| 33 | 2024-12-08 | To which country did Bashar al-Assad flee when his regime fell to rebel forces on 8 December 2024? | Russia / Russia | Russia | exact_alias_match | none recorded |
| 34 | 2025-01 | Which Chinese AI company&#x27;s R1 reasoning model triggered a US tech-stock selloff in January 2025? | DeepSeek / deepseek | DeepSeek | exact_alias_match | none recorded |
| 35 | 2025-01 | Which US city was devastated by the Palisades and Eaton wildfires in January 2025? | Los Angeles / los angeles | Los Angeles | exact_alias_match | none recorded |
| 36 | 2025-01 | Who was inaugurated as the 47th president of the United States in January 2025? | Donald Trump / trump | Donald Trump | exact_alias_match | earlier_election_result_predicts_identity |
| 37 | 2025-01-07 | What was the name of the largest of the wildfires that broke out in Los Angeles County on January 7, 2025? | The Palisades Fire / The Palisades Fire | Palisades Fire | manual_review | cross_probe_context_cue |
| 38 | 2025-01-29 | How many people died in the midair collision between an American Airlines regional jet and a US Army Black Hawk helicopter over the Potomac River near Washington DC on January 29, 2025? | 67 / 67 | 67 | exact_alias_match | none recorded |
| 39 | 2025-02 | Which team won Super Bowl LIX in February 2025? | Philadelphia Eagles / eagles, philadelphia | Philadelphia Eagles | exact_alias_match | none recorded |
| 40 | 2025-02 | Which party bloc won Germany&#x27;s February 2025 federal election? | CDU/CSU / cdu, union | CDU/CSU | exact_alias_match | none recorded |
| 41 | 2025-02 | Which Oscar-winning actor was found dead along with his wife at their New Mexico home in February 2025? | Gene Hackman / hackman | Gene Hackman | exact_alias_match | none recorded |
| 42 | 2025-02-24 | What hybrid reasoning model did Anthropic release on February 24, 2025, alongside the launch of Claude Code? | Claude 3.7 Sonnet / Claude 3.7 Sonnet | Claude 3.7 Sonnet | exact_alias_match | none recorded |
| 43 | 2025-02-27 | What model did OpenAI release as a research preview on February 27, 2025, described as its largest and last non-chain-of-thought model? | GPT-4.5 / GPT-4.5 | GPT-4.5 | exact_alias_match | release_does_not_establish_first_public_model_name |
| 44 | 2025-03 | Which film won Best Picture at the March 2025 Academy Awards? | Anora / anora | Anora | exact_alias_match | none recorded |
| 45 | 2025-03 | Who became Canada&#x27;s prime minister in March 2025? | Mark Carney / carney | Mark Carney | exact_alias_match | none recorded |
| 46 | 2025-03 | The editor-in-chief of which magazine was accidentally added to a Signal group chat of US officials discussing Yemen strikes in March 2025? | The Atlantic / atlantic | The Atlantic | exact_alias_match | none recorded |
| 47 | 2025-03 | Which country was struck by a devastating magnitude-7.7 earthquake in March 2025? | Myanmar / myanmar, burma | UNKNOWN | explicit_unknown | cross_probe_context_cue |
| 48 | 2025-03-25 | What AI model did Google announce on March 25, 2025, which debuted at #1 on LMArena? | Gemini 2.5 Pro / Gemini 2.5 Pro, Gemini 2.5 Pro (Experimental) | Gemini 2.5 Pro Experimental | manual_review | none recorded |
| 49 | 2025-03-28 | What was the magnitude of the earthquake that struck near Mandalay, Myanmar on March 28, 2025? | 7.7 / 7.7 | 7.7 magnitude | manual_review | cross_probe_context_cue |
| 50 | 2025-04 | Which pope died in April 2025? | Pope Francis / francis | Pope Francis | exact_alias_match | none recorded |
| 51 | 2025-04 | What name did Donald Trump give to his sweeping tariff announcement of April 2, 2025? | Liberation Day / liberation | Liberation Day | exact_alias_match | name_announced_before_tariff_event |
| 52 | 2025-04 | Which party won Canada&#x27;s April 2025 federal election? | Liberal Party / liberal | Liberal Party of Canada | manual_review | none recorded |
| 53 | 2025-05 | What papal name did Cardinal Robert Prevost take when elected pope in May 2025? | Leo XIV / leo | Leo XIV | exact_alias_match | ordinal_omitted_in_accepted_alias |
| 54 | 2025-05 | Which club won the 2024-25 UEFA Champions League final in May 2025? | Paris Saint-Germain / paris, psg | Paris Saint-Germain | exact_alias_match | city_only_club_alias |
| 55 | 2025-05 | Which party won Australia&#x27;s May 2025 federal election? | Labor / labor, albanese | Australian Labor Party | manual_review | person_alias_for_party_question |
| 56 | 2025-05 | Which two nuclear-armed neighbors fought a four-day military conflict in May 2025 following the Pahalgam attack? | India and Pakistan / pakistan | India and Pakistan | exact_alias_match | cross_probe_context_cue, one_country_alias_for_two_country_question |
| 57 | 2025-05-03 | Which horse won the 151st Kentucky Derby on May 3, 2025? | Sovereignty / Sovereignty | Sovereignty | exact_alias_match | none recorded |
| 58 | 2025-05-07 | What was the codename of India&#x27;s May 7, 2025 missile strikes on Pakistan, launched in response to the April 22 Pahalgam attack? | Operation Sindoor / Operation Sindoor | Operation Sindoor | exact_alias_match | none recorded |
| 59 | 2025-05-08 | Who was elected pope on May 8, 2025, becoming the first American pontiff, and what name did he take? | Cardinal Robert Francis Prevost, who took the name Leo XIV / Cardinal Robert Francis Prevost, who took the name Leo XIV | Robert Francis Prevost, Pope Leo XIV | manual_review | cross_probe_context_cue, compound_exact_alias_is_narrow |
| 60 | 2025-06 | Which Iranian nuclear site, buried deep inside a mountain, was the primary target of the June 2025 US B-2 bunker-buster strikes? | Fordow / fordow, fordo | UNKNOWN | explicit_unknown | primary_target_is_editorial_characterization, cross_probe_context_cue |
| 61 | 2025-06 | In which Indian city did an Air India Boeing 787 crash shortly after takeoff in June 2025? | Ahmedabad / ahmedabad | Ahmedabad | exact_alias_match | none recorded |
| 62 | 2025-06 | Which team won the 2025 NBA Finals? | Oklahoma City Thunder / thunder, oklahoma | Oklahoma City Thunder | exact_alias_match | none recorded |
| 63 | 2025-06-13 | What was the codename of Israel&#x27;s large-scale air campaign against Iran&#x27;s nuclear program and military leadership launched June 13, 2025? | Operation Rising Lion / Operation Rising Lion | Operation Rising Lion | exact_alias_match | none recorded |
| 64 | 2025-06-22 | What was the codename of the US B-2 bomber strike on Iran&#x27;s Fordow, Natanz, and Isfahan nuclear sites on the night of June 21-22, 2025? | Operation Midnight Hammer / Operation Midnight Hammer | Operation Midnight Hammer | exact_alias_match | none recorded |
| 65 | 2025-07 | Who won the 2025 Wimbledon men&#x27;s singles title? | Jannik Sinner / sinner | Jannik Sinner | exact_alias_match | none recorded |
| 66 | 2025-07 | Which heavy-metal singer died in July 2025, weeks after his farewell concert in Birmingham? | Ozzy Osbourne / osbourne, ozzy | Ozzy Osbourne | exact_alias_match | none recorded |
| 67 | 2025-07 | Which club won the expanded FIFA Club World Cup final in July 2025? | Chelsea / chelsea | Chelsea F.C. | manual_review | none recorded |
| 68 | 2025-07 | Which professional-wrestling icon died in July 2025 at age 71? | Hulk Hogan / hogan | Hulk Hogan | exact_alias_match | none recorded |
| 69 | 2025-07-27 | Which national team won the UEFA Women&#x27;s Euro 2025 final on July 27, 2025, beating Spain on penalties in Basel? | England / England | England | exact_alias_match | none recorded |
| 70 | 2025-08 | What flagship model did OpenAI release in August 2025? | GPT-5 / gpt-5, gpt5 | GPT-5 | exact_alias_match | none recorded |
| 71 | 2025-08 | In which US state did Donald Trump and Vladimir Putin hold their August 2025 summit? | Alaska / alaska | Alaska | exact_alias_match | none recorded |
| 72 | 2025-09 | Which conservative activist was assassinated at Utah Valley University in September 2025? | Charlie Kirk / kirk | Charlie Kirk | exact_alias_match | cross_probe_context_cue |
| 73 | 2025-09 | Gen-Z-led protests in September 2025 forced the prime minister of which Himalayan country to resign? | Nepal / nepal | Nepal | exact_alias_match | none recorded |
| 74 | 2025-09 | Who was appointed France&#x27;s prime minister in September 2025? | Sébastien Lecornu / lecornu | Sébastien Lecornu | exact_alias_match | none recorded |
| 75 | 2025-09 | Who won the 2025 US Open men&#x27;s singles title? | Carlos Alcaraz / alcaraz | Carlos Alcaraz | exact_alias_match | none recorded |
| 76 | 2025-09-10 | Conservative activist Charlie Kirk was assassinated at Utah Valley University on September 10, 2025. What is the name of the man charged with the shooting? | Tyler Robinson / Tyler Robinson | UNKNOWN | explicit_unknown | shooting_date_precedes_requested_suspect_fact |
| 77 | 2025-09-22 | Which PSG forward won the men&#x27;s 2025 Ballon d&#x27;Or, awarded on September 22, 2025? | Ousmane Dembélé / Ousmane Dembélé | Ousmane Dembélé | exact_alias_match | diacritic_sensitive_exact_alias |
| 78 | 2025-09-27 | Which national team won the 2025 Women&#x27;s Rugby World Cup final at Twickenham on September 27, 2025, beating Canada 33-13? | England / England | England | exact_alias_match | none recorded |
| 79 | 2025-09-29 | Which AI model did Anthropic release on September 29, 2025, billing it as the best coding model in the world alongside Claude Code 2.0? | Claude Sonnet 4.5 / Claude Sonnet 4.5 | Claude Sonnet 4.5 | exact_alias_match | none recorded |
| 80 | 2025-10 | Which Paris museum had crown jewels stolen in a daylight heist in October 2025? | Louvre / louvre | UNKNOWN | explicit_unknown | none recorded |
| 81 | 2025-10 | Who won the 2025 Nobel Peace Prize? | María Corina Machado / machado | UNKNOWN | explicit_unknown | none recorded |
| 82 | 2025-10 | Who became Japan&#x27;s first female prime minister in October 2025? | Sanae Takaichi / takaichi | Sanae Takaichi | exact_alias_match | none recorded |
| 83 | 2025-10 | Under whose 20-point peace plan did Israel and Hamas agree an October 2025 ceasefire and hostage release? | Donald Trump / trump | Donald Trump | exact_alias_match | none recorded |
| 84 | 2025-10-13 | On October 13, 2025, under the Gaza ceasefire deal, Hamas released all remaining living Israeli hostages. How many living hostages were released that day? | 20 / 20 | 20 | exact_alias_match | none recorded |
| 85 | 2025-10-28 | Hurricane Melissa made landfall in Jamaica on October 28, 2025 as the strongest hurricane ever recorded to hit the island. What Saffir-Simpson category was it at landfall? | Category 5 / Category 5 | Category 5 | exact_alias_match | none recorded |
| 86 | 2025-11 | Who won the November 2025 New York City mayoral election? | Zohran Mamdani / mamdani | Zohran Mamdani | exact_alias_match | none recorded |
| 87 | 2025-11 | Which team won the 2025 World Series? | Los Angeles Dodgers / dodgers | Los Angeles Dodgers | exact_alias_match | none recorded |
| 88 | 2025-11 | Which former US vice president died in November 2025? | Dick Cheney / cheney | Dick Cheney | exact_alias_match | none recorded |
| 89 | 2025-11 | Which Brazilian city hosted the COP30 climate summit in November 2025? | Belém / belem, belém | Belém | exact_alias_match | host_known_before_benchmark_date_range |
| 90 | 2025-11 | What was Anthropic&#x27;s flagship Claude model release of November 2025 called? | Claude Opus 4.5 / opus 4.5, 4.5 | Claude Opus 4.5 | exact_alias_match | none recorded |
| 91 | 2025-11-12 | The longest US federal government shutdown in history ended on November 12, 2025 when President Trump signed a funding bill. How many days did it last? | 43 / 43 | 43 days | manual_review | none recorded |
| 92 | 2025-11-18 | Which flagship AI model did Google release on November 18, 2025, topping the LMArena leaderboard and launching alongside the Antigravity IDE? | Gemini 3 / Gemini 3 | Gemini 3 Pro | manual_review | none recorded |
| 93 | 2025-12 | Who won Chile&#x27;s December 2025 presidential runoff? | José Antonio Kast / kast | UNKNOWN | explicit_unknown | none recorded |
| 94 | 2026-02 | Who won Super Bowl LX in February 2026? | Seattle Seahawks / seahawks, seattle | UNKNOWN | explicit_unknown | none recorded |
| 95 | 2026-02 | Which country topped the medal table at the 2026 Winter Olympics in Milan-Cortina? | Norway / norway | UNKNOWN | explicit_unknown | none recorded |
| 96 | 2026-02 | Which country won the gold medal in men&#x27;s ice hockey at the 2026 Winter Olympics? | United States / united states, usa, american | UNKNOWN | explicit_unknown | none recorded |
| 97 | 2026-02 | Who won Album of the Year at the Grammy Awards held in February 2026? | Bad Bunny (Debí Tirar Más Fotos) / bad bunny, debi tirar, debí tirar | Bad Bunny | exact_alias_match | none recorded |
| 98 | 2026-02 | Who won the men&#x27;s singles title at the 2026 Australian Open? | Carlos Alcaraz / alcaraz | UNKNOWN | explicit_unknown | none recorded |
| 99 | 2026-02 | Who won Costa Rica&#x27;s presidential election held on February 1, 2026? | Laura Fernández / fernandez, fernández | UNKNOWN | explicit_unknown | none recorded |
| 100 | 2026-02 | Which country&#x27;s supreme leader was assassinated in airstrikes on his capital on February 28, 2026? | Iran (Ali Khamenei) / iran, khamenei | UNKNOWN | explicit_unknown | person_alias_for_country_question |
| 101 | 2026-02-15 | Which Oscar-winning actor, star of The Godfather and Apocalypse Now, died on February 15, 2026 at age 95? | Robert Duvall / Robert Duvall | UNKNOWN | explicit_unknown | publication_after_event_day |
| 102 | 2026-03 | Which film won Best Picture at the 98th Academy Awards in March 2026? | One Battle After Another / one battle after another | UNKNOWN | explicit_unknown | none recorded |
| 103 | 2026-03 | Which country won the 2026 Six Nations rugby championship? | France / france | UNKNOWN | explicit_unknown | none recorded |
| 104 | 2026-03 | Which version of its flagship GPT model series did OpenAI release on March 5, 2026? | GPT-5.4 / 5.4 | GPT-5.4 | exact_alias_match | none recorded |
| 105 | 2026-03 | Who was elected Iran&#x27;s third supreme leader in March 2026? | Mojtaba Khamenei / mojtaba | UNKNOWN | explicit_unknown | none recorded |
| 106 | 2026-03-15 | Who won Best Actor at the 98th Academy Awards on March 15, 2026? | Michael B. Jordan / Michael B. Jordan, Michael B. Jordan (for Sinners) | UNKNOWN | explicit_unknown | none recorded |
| 107 | 2026-04 | Who became Prime Minister of Hungary following the April 2026 parliamentary election? | Péter Magyar / magyar | UNKNOWN | explicit_unknown | election_month_is_not_taking_office_month |
| 108 | 2026-04 | Which company released the AI model GPT-5.5 in April 2026? | OpenAI / openai | UNKNOWN | explicit_unknown | brand_to_company_inference |
| 109 | 2026-04-06 | Which team won the 2026 NCAA men&#x27;s basketball national championship on April 6, 2026? | Michigan Wolverines / Michigan Wolverines, Michigan Wolverines (beat UConn 69-63) | UNKNOWN | explicit_unknown | none recorded |
| 110 | 2026-05 | Which horse won the 2026 Kentucky Derby? | Golden Tempo / golden tempo | UNKNOWN | explicit_unknown | none recorded |
| 111 | 2026-05 | Which country won the Eurovision Song Contest 2026? | Bulgaria / bulgaria | UNKNOWN | explicit_unknown | none recorded |
| 112 | 2026-05 | Which film won the Palme d&#x27;Or at the 2026 Cannes Film Festival? | Fjord / fjord, mungiu | UNKNOWN | explicit_unknown | director_alias_for_film_question |
| 113 | 2026-05 | Which club won the 2026 UEFA Champions League final in Budapest? | Paris Saint-Germain / paris, psg | UNKNOWN | explicit_unknown | city_only_club_alias |
| 114 | 2026-06 | Who won the men&#x27;s singles title at the 2026 French Open? | Alexander Zverev / zverev | UNKNOWN | explicit_unknown | none recorded |
| 115 | 2026-06 | Which team won the 2026 NBA Finals? | New York Knicks / knicks | UNKNOWN | explicit_unknown | none recorded |
| 116 | 2026-06 | Which team won the 2026 Stanley Cup Final? | Carolina Hurricanes / hurricanes, carolina | UNKNOWN | explicit_unknown | none recorded |
| 117 | 2026-06 | Who won Colombia&#x27;s presidential runoff election on June 21, 2026? | Abelardo de la Espriella / espriella, abelardo | UNKNOWN | explicit_unknown | original_source_is_preliminary_result |
| 118 | 2026-06-06 | Who won the 2026 French Open women&#x27;s singles title on June 6, 2026? | Mirra Andreeva / Mirra Andreeva, Mirra Andreeva (beat Maja Chwalinska) | UNKNOWN | explicit_unknown | none recorded |
| 119 | 2026-06-09 | Which flagship AI model did Anthropic release on June 9, 2026, alongside the restricted Mythos 5? | Claude Fable 5 / Claude Fable 5 | UNKNOWN | explicit_unknown | none recorded |
| 120 | 2026-06-24 | Which country was struck by twin magnitude 7.2 and 7.5 earthquakes on June 24, 2026, its strongest since 1900, killing over 2,000 people? | Venezuela / Venezuela | UNKNOWN | explicit_unknown | time_varying_casualty_premise |
| 121 | 2026-07 | Which country won the 2026 FIFA World Cup final on July 19, 2026? | Spain / spain | UNKNOWN | explicit_unknown | none recorded |
| 122 | 2026-07 | Who won the men&#x27;s singles title at Wimbledon 2026? | Jannik Sinner / sinner | UNKNOWN | explicit_unknown | none recorded |
| 123 | 2026-07 | Who won the women&#x27;s singles title at Wimbledon 2026? | Linda Nosková / noskova, nosková | UNKNOWN | explicit_unknown | none recorded |
| 124 | 2026-07 | Who won the 2026 Tour de France general classification? | Tadej Pogačar / pogacar, pogačar | UNKNOWN | explicit_unknown | none recorded |
| 125 | 2026-07 | Who won the 2026 Open Championship (British Open) at Royal Birkdale? | Ryan Fox / ryan fox | UNKNOWN | explicit_unknown | none recorded |
| 126 | 2026-07 | Which company released the AI model Claude Opus 5 on July 24, 2026? | Anthropic / anthropic | UNKNOWN | explicit_unknown | brand_to_company_inference |
| 127 | 2026-07 | Which version of its flagship GPT model series did OpenAI release on July 9, 2026? | GPT-5.6 / 5.6 | UNKNOWN | explicit_unknown | limited_preview_precedes_public_release |
| 128 | 2026-08 | Which country topped the medal table at the 2026 Commonwealth Games in Glasgow? | Australia / australia | UNKNOWN | explicit_unknown | none recorded |
| 129 | 2026-08 | Which actress, star of Heroes and Nashville, died on August 16, 2026, at age 36? | Hayden Panettiere / panettiere | UNKNOWN | explicit_unknown | none recorded |

</details>

<details><summary>gpt-5.6-luna · medium · all 129 probes</summary>

| # | Recorded event date | Question | Original answer / aliases | Full response | Original-key grade | Source flags |
| ---: | --- | --- | --- | --- | --- | --- |
| 1 | 2024-07 | Who became the Democratic Party&#x27;s presidential nominee after Joe Biden withdrew from the 2024 race? | Kamala Harris / harris | Kamala Harris | exact_alias_match | nominee_stage_date_mismatch |
| 2 | 2024-07 | In which US state did the July 2024 assassination attempt on Donald Trump take place? | Pennsylvania / pennsylvania | Pennsylvania | exact_alias_match | cross_probe_context_cue |
| 3 | 2024-07 | Which party won the July 2024 United Kingdom general election? | Labour / labour, labor | Labour Party | manual_review | cross_probe_context_cue |
| 4 | 2024-07 | A faulty update from which cybersecurity company caused a global Windows outage in July 2024? | CrowdStrike / crowdstrike | CrowdStrike | exact_alias_match | none recorded |
| 5 | 2024-07 | Which national team won UEFA Euro 2024? | Spain / spain | Spain | exact_alias_match | none recorded |
| 6 | 2024-07 | Which national team won the 2024 Copa América? | Argentina / argentina | Argentina | exact_alias_match | none recorded |
| 7 | 2024-07-05 | Who became UK Prime Minister on 5 July 2024 after Labour&#x27;s landslide win in the July 4 general election? | Keir Starmer / Keir Starmer | Keir Starmer | exact_alias_match | none recorded |
| 8 | 2024-07-13 | In which Pennsylvania town was Donald Trump wounded in an assassination attempt at a campaign rally on 13 July 2024? | Butler / Butler | Butler | exact_alias_match | near_city_not_within_city |
| 9 | 2024-08 | Which country&#x27;s prime minister, Sheikh Hasina, resigned and fled the country in August 2024? | Bangladesh / bangladesh | Bangladesh | exact_alias_match | none recorded |
| 10 | 2024-08 | Which country won the most total medals at the Paris 2024 Olympics? | United States / united states, usa, u.s. | United States | exact_alias_match | none recorded |
| 11 | 2024-08-04 | Who won the men&#x27;s 100m final at the Paris 2024 Olympics on 4 August 2024, in a photo finish over Kishane Thompson? | Noah Lyles / Noah Lyles | Noah Lyles | exact_alias_match | none recorded |
| 12 | 2024-08-11 | How many gold medals did both the United States and China each finish with atop the Paris 2024 Olympics medal table (closing 11 August 2024)? | 40 / 40 | 40 gold medals | manual_review | none recorded |
| 13 | 2024-08-18 | Which French film legend, star of &#x27;Le Samouraï&#x27; and &#x27;Purple Noon&#x27;, died on 18 August 2024 at age 88? | Alain Delon / Alain Delon | Alain Delon | exact_alias_match | none recorded |
| 14 | 2024-09 | In which country did thousands of pagers used by Hezbollah members explode in September 2024? | Lebanon / lebanon | Lebanon | exact_alias_match | geographic_scope_not_exclusive |
| 15 | 2024-09 | Which Hezbollah secretary-general was killed in an Israeli airstrike in September 2024? | Hassan Nasrallah / nasrallah | Hassan Nasrallah | exact_alias_match | none recorded |
| 16 | 2024-09-12 | What was the name of the reasoning model family OpenAI released in preview on 12 September 2024? | o1 / o1, o1 (o1-preview) | o1-preview | manual_review | none recorded |
| 17 | 2024-09-26 | At what Saffir-Simpson category did Hurricane Helene make landfall in Florida&#x27;s Big Bend on 26 September 2024? | Category 4 / Category 4 | Category 4 | exact_alias_match | none recorded |
| 18 | 2024-10 | Which organization won the 2024 Nobel Peace Prize? | Nihon Hidankyo / hidankyo | Nihon Hidankyo | exact_alias_match | none recorded |
| 19 | 2024-10 | Which team won the 2024 World Series? | Los Angeles Dodgers / dodgers | Los Angeles Dodgers | exact_alias_match | none recorded |
| 20 | 2024-10-08 | Which two scientists shared the 2024 Nobel Prize in Physics (announced 8 October 2024) for foundational work on machine learning with artificial neural networks? | John Hopfield and Geoffrey Hinton / John Hopfield and Geoffrey Hinton | John Hopfield and Geoffrey Hinton | exact_alias_match | compound_exact_alias_is_narrow |
| 21 | 2024-10-09 | At what Saffir-Simpson category did Hurricane Milton make landfall near Siesta Key, Florida on 9 October 2024? | Category 3 / Category 3 | Category 3 | exact_alias_match | none recorded |
| 22 | 2024-10-22 | What new capability, letting the model control a computer&#x27;s screen, cursor and keyboard, did Anthropic announce in public beta alongside the upgraded Claude 3.5 Sonnet on 22 October 2024? | Computer use / Computer use | Computer use | exact_alias_match | answer_inferable_from_definition |
| 23 | 2024-11 | Who won the 2024 United States presidential election? | Donald Trump / trump | Donald Trump | exact_alias_match | none recorded |
| 24 | 2024-11 | Who won the November 2024 boxing match between Jake Paul and Mike Tyson? | Jake Paul / paul | Jake Paul | exact_alias_match | named_candidate_choice |
| 25 | 2024-11-05 | Which party won control of the US Senate in the 5 November 2024 elections, and with how many seats? | Republicans, 53 seats / Republicans, 53 seats | Republican Party, 53 seats | manual_review | compound_exact_alias_is_narrow |
| 26 | 2024-12 | Which Syrian president was overthrown when rebels took Damascus in December 2024? | Bashar al-Assad / assad | Bashar al-Assad | exact_alias_match | none recorded |
| 27 | 2024-12 | Which country&#x27;s president declared a short-lived martial law in December 2024? | South Korea / south korea | South Korea | exact_alias_match | none recorded |
| 28 | 2024-12 | Which former US president died on December 29, 2024? | Jimmy Carter / carter | Jimmy Carter | exact_alias_match | none recorded |
| 29 | 2024-12 | The CEO of which health-insurance company was fatally shot in Manhattan in December 2024? | UnitedHealthcare / unitedhealth | UnitedHealthcare | exact_alias_match | none recorded |
| 30 | 2024-12 | Who won the 2024 World Chess Championship? | Gukesh Dommaraju / gukesh | D. Gukesh | manual_review | none recorded |
| 31 | 2024-12 | Which Paris cathedral reopened in December 2024 after restoration from its 2019 fire? | Notre-Dame / notre | Notre-Dame de Paris | manual_review | none recorded |
| 32 | 2024-12-03 | Which South Korean president declared emergency martial law on 3 December 2024, rescinded within about six hours? | Yoon Suk Yeol / Yoon Suk Yeol | Yoon Suk Yeol | exact_alias_match | none recorded |
| 33 | 2024-12-08 | To which country did Bashar al-Assad flee when his regime fell to rebel forces on 8 December 2024? | Russia / Russia | Russia | exact_alias_match | none recorded |
| 34 | 2025-01 | Which Chinese AI company&#x27;s R1 reasoning model triggered a US tech-stock selloff in January 2025? | DeepSeek / deepseek | DeepSeek | exact_alias_match | none recorded |
| 35 | 2025-01 | Which US city was devastated by the Palisades and Eaton wildfires in January 2025? | Los Angeles / los angeles | Los Angeles | exact_alias_match | none recorded |
| 36 | 2025-01 | Who was inaugurated as the 47th president of the United States in January 2025? | Donald Trump / trump | Donald Trump | exact_alias_match | earlier_election_result_predicts_identity |
| 37 | 2025-01-07 | What was the name of the largest of the wildfires that broke out in Los Angeles County on January 7, 2025? | The Palisades Fire / The Palisades Fire | Palisades Fire | manual_review | cross_probe_context_cue |
| 38 | 2025-01-29 | How many people died in the midair collision between an American Airlines regional jet and a US Army Black Hawk helicopter over the Potomac River near Washington DC on January 29, 2025? | 67 / 67 | 67 | exact_alias_match | none recorded |
| 39 | 2025-02 | Which team won Super Bowl LIX in February 2025? | Philadelphia Eagles / eagles, philadelphia | Philadelphia Eagles | exact_alias_match | none recorded |
| 40 | 2025-02 | Which party bloc won Germany&#x27;s February 2025 federal election? | CDU/CSU / cdu, union | CDU/CSU | exact_alias_match | none recorded |
| 41 | 2025-02 | Which Oscar-winning actor was found dead along with his wife at their New Mexico home in February 2025? | Gene Hackman / hackman | Gene Hackman | exact_alias_match | none recorded |
| 42 | 2025-02-24 | What hybrid reasoning model did Anthropic release on February 24, 2025, alongside the launch of Claude Code? | Claude 3.7 Sonnet / Claude 3.7 Sonnet | Claude 3.7 Sonnet | exact_alias_match | none recorded |
| 43 | 2025-02-27 | What model did OpenAI release as a research preview on February 27, 2025, described as its largest and last non-chain-of-thought model? | GPT-4.5 / GPT-4.5 | GPT-4.5 | exact_alias_match | release_does_not_establish_first_public_model_name |
| 44 | 2025-03 | Which film won Best Picture at the March 2025 Academy Awards? | Anora / anora | Anora | exact_alias_match | none recorded |
| 45 | 2025-03 | Who became Canada&#x27;s prime minister in March 2025? | Mark Carney / carney | Mark Carney | exact_alias_match | none recorded |
| 46 | 2025-03 | The editor-in-chief of which magazine was accidentally added to a Signal group chat of US officials discussing Yemen strikes in March 2025? | The Atlantic / atlantic | The Atlantic | exact_alias_match | none recorded |
| 47 | 2025-03 | Which country was struck by a devastating magnitude-7.7 earthquake in March 2025? | Myanmar / myanmar, burma | Myanmar | exact_alias_match | cross_probe_context_cue |
| 48 | 2025-03-25 | What AI model did Google announce on March 25, 2025, which debuted at #1 on LMArena? | Gemini 2.5 Pro / Gemini 2.5 Pro, Gemini 2.5 Pro (Experimental) | Gemini 2.5 Pro | exact_alias_match | none recorded |
| 49 | 2025-03-28 | What was the magnitude of the earthquake that struck near Mandalay, Myanmar on March 28, 2025? | 7.7 / 7.7 | 7.7 magnitude | manual_review | cross_probe_context_cue |
| 50 | 2025-04 | Which pope died in April 2025? | Pope Francis / francis | Pope Francis | exact_alias_match | none recorded |
| 51 | 2025-04 | What name did Donald Trump give to his sweeping tariff announcement of April 2, 2025? | Liberation Day / liberation | “Liberation Day” | exact_alias_match | name_announced_before_tariff_event |
| 52 | 2025-04 | Which party won Canada&#x27;s April 2025 federal election? | Liberal Party / liberal | Liberal Party of Canada | manual_review | none recorded |
| 53 | 2025-05 | What papal name did Cardinal Robert Prevost take when elected pope in May 2025? | Leo XIV / leo | Leo XIV | exact_alias_match | ordinal_omitted_in_accepted_alias |
| 54 | 2025-05 | Which club won the 2024-25 UEFA Champions League final in May 2025? | Paris Saint-Germain / paris, psg | Paris Saint-Germain | exact_alias_match | city_only_club_alias |
| 55 | 2025-05 | Which party won Australia&#x27;s May 2025 federal election? | Labor / labor, albanese | Australian Labor Party | manual_review | person_alias_for_party_question |
| 56 | 2025-05 | Which two nuclear-armed neighbors fought a four-day military conflict in May 2025 following the Pahalgam attack? | India and Pakistan / pakistan | India and Pakistan | exact_alias_match | cross_probe_context_cue, one_country_alias_for_two_country_question |
| 57 | 2025-05-03 | Which horse won the 151st Kentucky Derby on May 3, 2025? | Sovereignty / Sovereignty | Sovereignty | exact_alias_match | none recorded |
| 58 | 2025-05-07 | What was the codename of India&#x27;s May 7, 2025 missile strikes on Pakistan, launched in response to the April 22 Pahalgam attack? | Operation Sindoor / Operation Sindoor | Operation Sindoor | exact_alias_match | none recorded |
| 59 | 2025-05-08 | Who was elected pope on May 8, 2025, becoming the first American pontiff, and what name did he take? | Cardinal Robert Francis Prevost, who took the name Leo XIV / Cardinal Robert Francis Prevost, who took the name Leo XIV | Robert Francis Prevost, Pope Leo XIV | manual_review | cross_probe_context_cue, compound_exact_alias_is_narrow |
| 60 | 2025-06 | Which Iranian nuclear site, buried deep inside a mountain, was the primary target of the June 2025 US B-2 bunker-buster strikes? | Fordow / fordow, fordo | Fordow Fuel Enrichment Plant | manual_review | primary_target_is_editorial_characterization, cross_probe_context_cue |
| 61 | 2025-06 | In which Indian city did an Air India Boeing 787 crash shortly after takeoff in June 2025? | Ahmedabad / ahmedabad | Ahmedabad | exact_alias_match | none recorded |
| 62 | 2025-06 | Which team won the 2025 NBA Finals? | Oklahoma City Thunder / thunder, oklahoma | Oklahoma City Thunder | exact_alias_match | none recorded |
| 63 | 2025-06-13 | What was the codename of Israel&#x27;s large-scale air campaign against Iran&#x27;s nuclear program and military leadership launched June 13, 2025? | Operation Rising Lion / Operation Rising Lion | Operation Rising Lion | exact_alias_match | none recorded |
| 64 | 2025-06-22 | What was the codename of the US B-2 bomber strike on Iran&#x27;s Fordow, Natanz, and Isfahan nuclear sites on the night of June 21-22, 2025? | Operation Midnight Hammer / Operation Midnight Hammer | Operation Midnight Hammer | exact_alias_match | none recorded |
| 65 | 2025-07 | Who won the 2025 Wimbledon men&#x27;s singles title? | Jannik Sinner / sinner | Jannik Sinner | exact_alias_match | none recorded |
| 66 | 2025-07 | Which heavy-metal singer died in July 2025, weeks after his farewell concert in Birmingham? | Ozzy Osbourne / osbourne, ozzy | Ozzy Osbourne | exact_alias_match | none recorded |
| 67 | 2025-07 | Which club won the expanded FIFA Club World Cup final in July 2025? | Chelsea / chelsea | Chelsea FC | manual_review | none recorded |
| 68 | 2025-07 | Which professional-wrestling icon died in July 2025 at age 71? | Hulk Hogan / hogan | Hulk Hogan | exact_alias_match | none recorded |
| 69 | 2025-07-27 | Which national team won the UEFA Women&#x27;s Euro 2025 final on July 27, 2025, beating Spain on penalties in Basel? | England / England | England | exact_alias_match | none recorded |
| 70 | 2025-08 | What flagship model did OpenAI release in August 2025? | GPT-5 / gpt-5, gpt5 | GPT-5 | exact_alias_match | none recorded |
| 71 | 2025-08 | In which US state did Donald Trump and Vladimir Putin hold their August 2025 summit? | Alaska / alaska | Alaska | exact_alias_match | none recorded |
| 72 | 2025-09 | Which conservative activist was assassinated at Utah Valley University in September 2025? | Charlie Kirk / kirk | Charlie Kirk | exact_alias_match | cross_probe_context_cue |
| 73 | 2025-09 | Gen-Z-led protests in September 2025 forced the prime minister of which Himalayan country to resign? | Nepal / nepal | Nepal | exact_alias_match | none recorded |
| 74 | 2025-09 | Who was appointed France&#x27;s prime minister in September 2025? | Sébastien Lecornu / lecornu | Sébastien Lecornu | exact_alias_match | none recorded |
| 75 | 2025-09 | Who won the 2025 US Open men&#x27;s singles title? | Carlos Alcaraz / alcaraz | UNKNOWN | explicit_unknown | none recorded |
| 76 | 2025-09-10 | Conservative activist Charlie Kirk was assassinated at Utah Valley University on September 10, 2025. What is the name of the man charged with the shooting? | Tyler Robinson / Tyler Robinson | Tyler Robinson | exact_alias_match | shooting_date_precedes_requested_suspect_fact |
| 77 | 2025-09-22 | Which PSG forward won the men&#x27;s 2025 Ballon d&#x27;Or, awarded on September 22, 2025? | Ousmane Dembélé / Ousmane Dembélé | Ousmane Dembélé | exact_alias_match | diacritic_sensitive_exact_alias |
| 78 | 2025-09-27 | Which national team won the 2025 Women&#x27;s Rugby World Cup final at Twickenham on September 27, 2025, beating Canada 33-13? | England / England | England | exact_alias_match | none recorded |
| 79 | 2025-09-29 | Which AI model did Anthropic release on September 29, 2025, billing it as the best coding model in the world alongside Claude Code 2.0? | Claude Sonnet 4.5 / Claude Sonnet 4.5 | Claude Sonnet 4.5 | exact_alias_match | none recorded |
| 80 | 2025-10 | Which Paris museum had crown jewels stolen in a daylight heist in October 2025? | Louvre / louvre | Louvre Museum | manual_review | none recorded |
| 81 | 2025-10 | Who won the 2025 Nobel Peace Prize? | María Corina Machado / machado | Marielle Franco | manual_review | none recorded |
| 82 | 2025-10 | Who became Japan&#x27;s first female prime minister in October 2025? | Sanae Takaichi / takaichi | Sanae Takaichi | exact_alias_match | none recorded |
| 83 | 2025-10 | Under whose 20-point peace plan did Israel and Hamas agree an October 2025 ceasefire and hostage release? | Donald Trump / trump | Donald Trump’s | manual_review | none recorded |
| 84 | 2025-10-13 | On October 13, 2025, under the Gaza ceasefire deal, Hamas released all remaining living Israeli hostages. How many living hostages were released that day? | 20 / 20 | 20 | exact_alias_match | none recorded |
| 85 | 2025-10-28 | Hurricane Melissa made landfall in Jamaica on October 28, 2025 as the strongest hurricane ever recorded to hit the island. What Saffir-Simpson category was it at landfall? | Category 5 / Category 5 | Category 5 | exact_alias_match | none recorded |
| 86 | 2025-11 | Who won the November 2025 New York City mayoral election? | Zohran Mamdani / mamdani | Zohran Mamdani | exact_alias_match | none recorded |
| 87 | 2025-11 | Which team won the 2025 World Series? | Los Angeles Dodgers / dodgers | Los Angeles Dodgers | exact_alias_match | none recorded |
| 88 | 2025-11 | Which former US vice president died in November 2025? | Dick Cheney / cheney | Dick Cheney | exact_alias_match | none recorded |
| 89 | 2025-11 | Which Brazilian city hosted the COP30 climate summit in November 2025? | Belém / belem, belém | Belém | exact_alias_match | host_known_before_benchmark_date_range |
| 90 | 2025-11 | What was Anthropic&#x27;s flagship Claude model release of November 2025 called? | Claude Opus 4.5 / opus 4.5, 4.5 | Claude Opus 4.5 | exact_alias_match | none recorded |
| 91 | 2025-11-12 | The longest US federal government shutdown in history ended on November 12, 2025 when President Trump signed a funding bill. How many days did it last? | 43 / 43 | 43 days | manual_review | none recorded |
| 92 | 2025-11-18 | Which flagship AI model did Google release on November 18, 2025, topping the LMArena leaderboard and launching alongside the Antigravity IDE? | Gemini 3 / Gemini 3 | Gemini 3 Pro | manual_review | none recorded |
| 93 | 2025-12 | Who won Chile&#x27;s December 2025 presidential runoff? | José Antonio Kast / kast | José Antonio Kast | exact_alias_match | none recorded |
| 94 | 2026-02 | Who won Super Bowl LX in February 2026? | Seattle Seahawks / seahawks, seattle | UNKNOWN | explicit_unknown | none recorded |
| 95 | 2026-02 | Which country topped the medal table at the 2026 Winter Olympics in Milan-Cortina? | Norway / norway | UNKNOWN | explicit_unknown | none recorded |
| 96 | 2026-02 | Which country won the gold medal in men&#x27;s ice hockey at the 2026 Winter Olympics? | United States / united states, usa, american | UNKNOWN | explicit_unknown | none recorded |
| 97 | 2026-02 | Who won Album of the Year at the Grammy Awards held in February 2026? | Bad Bunny (Debí Tirar Más Fotos) / bad bunny, debi tirar, debí tirar | UNKNOWN | explicit_unknown | none recorded |
| 98 | 2026-02 | Who won the men&#x27;s singles title at the 2026 Australian Open? | Carlos Alcaraz / alcaraz | UNKNOWN | explicit_unknown | none recorded |
| 99 | 2026-02 | Who won Costa Rica&#x27;s presidential election held on February 1, 2026? | Laura Fernández / fernandez, fernández | UNKNOWN | explicit_unknown | none recorded |
| 100 | 2026-02 | Which country&#x27;s supreme leader was assassinated in airstrikes on his capital on February 28, 2026? | Iran (Ali Khamenei) / iran, khamenei | Iran | exact_alias_match | person_alias_for_country_question |
| 101 | 2026-02-15 | Which Oscar-winning actor, star of The Godfather and Apocalypse Now, died on February 15, 2026 at age 95? | Robert Duvall / Robert Duvall | Robert Duvall | exact_alias_match | publication_after_event_day |
| 102 | 2026-03 | Which film won Best Picture at the 98th Academy Awards in March 2026? | One Battle After Another / one battle after another | One Battle After Another | exact_alias_match | none recorded |
| 103 | 2026-03 | Which country won the 2026 Six Nations rugby championship? | France / france | UNKNOWN | explicit_unknown | none recorded |
| 104 | 2026-03 | Which version of its flagship GPT model series did OpenAI release on March 5, 2026? | GPT-5.4 / 5.4 | UNKNOWN | explicit_unknown | none recorded |
| 105 | 2026-03 | Who was elected Iran&#x27;s third supreme leader in March 2026? | Mojtaba Khamenei / mojtaba | UNKNOWN | explicit_unknown | none recorded |
| 106 | 2026-03-15 | Who won Best Actor at the 98th Academy Awards on March 15, 2026? | Michael B. Jordan / Michael B. Jordan, Michael B. Jordan (for Sinners) | Michael B. Jordan | exact_alias_match | none recorded |
| 107 | 2026-04 | Who became Prime Minister of Hungary following the April 2026 parliamentary election? | Péter Magyar / magyar | UNKNOWN | explicit_unknown | election_month_is_not_taking_office_month |
| 108 | 2026-04 | Which company released the AI model GPT-5.5 in April 2026? | OpenAI / openai | UNKNOWN | explicit_unknown | brand_to_company_inference |
| 109 | 2026-04-06 | Which team won the 2026 NCAA men&#x27;s basketball national championship on April 6, 2026? | Michigan Wolverines / Michigan Wolverines, Michigan Wolverines (beat UConn 69-63) | UNKNOWN | explicit_unknown | none recorded |
| 110 | 2026-05 | Which horse won the 2026 Kentucky Derby? | Golden Tempo / golden tempo | UNKNOWN | explicit_unknown | none recorded |
| 111 | 2026-05 | Which country won the Eurovision Song Contest 2026? | Bulgaria / bulgaria | UNKNOWN | explicit_unknown | none recorded |
| 112 | 2026-05 | Which film won the Palme d&#x27;Or at the 2026 Cannes Film Festival? | Fjord / fjord, mungiu | UNKNOWN | explicit_unknown | director_alias_for_film_question |
| 113 | 2026-05 | Which club won the 2026 UEFA Champions League final in Budapest? | Paris Saint-Germain / paris, psg | UNKNOWN | explicit_unknown | city_only_club_alias |
| 114 | 2026-06 | Who won the men&#x27;s singles title at the 2026 French Open? | Alexander Zverev / zverev | UNKNOWN | explicit_unknown | none recorded |
| 115 | 2026-06 | Which team won the 2026 NBA Finals? | New York Knicks / knicks | UNKNOWN | explicit_unknown | none recorded |
| 116 | 2026-06 | Which team won the 2026 Stanley Cup Final? | Carolina Hurricanes / hurricanes, carolina | UNKNOWN | explicit_unknown | none recorded |
| 117 | 2026-06 | Who won Colombia&#x27;s presidential runoff election on June 21, 2026? | Abelardo de la Espriella / espriella, abelardo | UNKNOWN | explicit_unknown | original_source_is_preliminary_result |
| 118 | 2026-06-06 | Who won the 2026 French Open women&#x27;s singles title on June 6, 2026? | Mirra Andreeva / Mirra Andreeva, Mirra Andreeva (beat Maja Chwalinska) | UNKNOWN | explicit_unknown | none recorded |
| 119 | 2026-06-09 | Which flagship AI model did Anthropic release on June 9, 2026, alongside the restricted Mythos 5? | Claude Fable 5 / Claude Fable 5 | UNKNOWN | explicit_unknown | none recorded |
| 120 | 2026-06-24 | Which country was struck by twin magnitude 7.2 and 7.5 earthquakes on June 24, 2026, its strongest since 1900, killing over 2,000 people? | Venezuela / Venezuela | UNKNOWN | explicit_unknown | time_varying_casualty_premise |
| 121 | 2026-07 | Which country won the 2026 FIFA World Cup final on July 19, 2026? | Spain / spain | UNKNOWN | explicit_unknown | none recorded |
| 122 | 2026-07 | Who won the men&#x27;s singles title at Wimbledon 2026? | Jannik Sinner / sinner | UNKNOWN | explicit_unknown | none recorded |
| 123 | 2026-07 | Who won the women&#x27;s singles title at Wimbledon 2026? | Linda Nosková / noskova, nosková | UNKNOWN | explicit_unknown | none recorded |
| 124 | 2026-07 | Who won the 2026 Tour de France general classification? | Tadej Pogačar / pogacar, pogačar | UNKNOWN | explicit_unknown | none recorded |
| 125 | 2026-07 | Who won the 2026 Open Championship (British Open) at Royal Birkdale? | Ryan Fox / ryan fox | UNKNOWN | explicit_unknown | none recorded |
| 126 | 2026-07 | Which company released the AI model Claude Opus 5 on July 24, 2026? | Anthropic / anthropic | UNKNOWN | explicit_unknown | brand_to_company_inference |
| 127 | 2026-07 | Which version of its flagship GPT model series did OpenAI release on July 9, 2026? | GPT-5.6 / 5.6 | UNKNOWN | explicit_unknown | limited_preview_precedes_public_release |
| 128 | 2026-08 | Which country topped the medal table at the 2026 Commonwealth Games in Glasgow? | Australia / australia | UNKNOWN | explicit_unknown | none recorded |
| 129 | 2026-08 | Which actress, star of Heroes and Nashville, died on August 16, 2026, at age 36? | Hayden Panettiere / panettiere | UNKNOWN | explicit_unknown | none recorded |

</details>

<details><summary>gpt-5.6-luna · high · all 129 probes</summary>

| # | Recorded event date | Question | Original answer / aliases | Full response | Original-key grade | Source flags |
| ---: | --- | --- | --- | --- | --- | --- |
| 1 | 2024-07 | Who became the Democratic Party&#x27;s presidential nominee after Joe Biden withdrew from the 2024 race? | Kamala Harris / harris | Kamala Harris | exact_alias_match | nominee_stage_date_mismatch |
| 2 | 2024-07 | In which US state did the July 2024 assassination attempt on Donald Trump take place? | Pennsylvania / pennsylvania | Pennsylvania | exact_alias_match | cross_probe_context_cue |
| 3 | 2024-07 | Which party won the July 2024 United Kingdom general election? | Labour / labour, labor | Labour Party | manual_review | cross_probe_context_cue |
| 4 | 2024-07 | A faulty update from which cybersecurity company caused a global Windows outage in July 2024? | CrowdStrike / crowdstrike | CrowdStrike | exact_alias_match | none recorded |
| 5 | 2024-07 | Which national team won UEFA Euro 2024? | Spain / spain | Spain | exact_alias_match | none recorded |
| 6 | 2024-07 | Which national team won the 2024 Copa América? | Argentina / argentina | Argentina | exact_alias_match | none recorded |
| 7 | 2024-07-05 | Who became UK Prime Minister on 5 July 2024 after Labour&#x27;s landslide win in the July 4 general election? | Keir Starmer / Keir Starmer | Keir Starmer | exact_alias_match | none recorded |
| 8 | 2024-07-13 | In which Pennsylvania town was Donald Trump wounded in an assassination attempt at a campaign rally on 13 July 2024? | Butler / Butler | Butler | exact_alias_match | near_city_not_within_city |
| 9 | 2024-08 | Which country&#x27;s prime minister, Sheikh Hasina, resigned and fled the country in August 2024? | Bangladesh / bangladesh | Bangladesh | exact_alias_match | none recorded |
| 10 | 2024-08 | Which country won the most total medals at the Paris 2024 Olympics? | United States / united states, usa, u.s. | United States | exact_alias_match | none recorded |
| 11 | 2024-08-04 | Who won the men&#x27;s 100m final at the Paris 2024 Olympics on 4 August 2024, in a photo finish over Kishane Thompson? | Noah Lyles / Noah Lyles | Noah Lyles | exact_alias_match | none recorded |
| 12 | 2024-08-11 | How many gold medals did both the United States and China each finish with atop the Paris 2024 Olympics medal table (closing 11 August 2024)? | 40 / 40 | 40 gold medals | manual_review | none recorded |
| 13 | 2024-08-18 | Which French film legend, star of &#x27;Le Samouraï&#x27; and &#x27;Purple Noon&#x27;, died on 18 August 2024 at age 88? | Alain Delon / Alain Delon | Alain Delon | exact_alias_match | none recorded |
| 14 | 2024-09 | In which country did thousands of pagers used by Hezbollah members explode in September 2024? | Lebanon / lebanon | Lebanon | exact_alias_match | geographic_scope_not_exclusive |
| 15 | 2024-09 | Which Hezbollah secretary-general was killed in an Israeli airstrike in September 2024? | Hassan Nasrallah / nasrallah | Hassan Nasrallah | exact_alias_match | none recorded |
| 16 | 2024-09-12 | What was the name of the reasoning model family OpenAI released in preview on 12 September 2024? | o1 / o1, o1 (o1-preview) | o1 | exact_alias_match | none recorded |
| 17 | 2024-09-26 | At what Saffir-Simpson category did Hurricane Helene make landfall in Florida&#x27;s Big Bend on 26 September 2024? | Category 4 / Category 4 | Category 4 | exact_alias_match | none recorded |
| 18 | 2024-10 | Which organization won the 2024 Nobel Peace Prize? | Nihon Hidankyo / hidankyo | Nihon Hidankyo | exact_alias_match | none recorded |
| 19 | 2024-10 | Which team won the 2024 World Series? | Los Angeles Dodgers / dodgers | Los Angeles Dodgers | exact_alias_match | none recorded |
| 20 | 2024-10-08 | Which two scientists shared the 2024 Nobel Prize in Physics (announced 8 October 2024) for foundational work on machine learning with artificial neural networks? | John Hopfield and Geoffrey Hinton / John Hopfield and Geoffrey Hinton | John J. Hopfield and Geoffrey E. Hinton | manual_review | compound_exact_alias_is_narrow |
| 21 | 2024-10-09 | At what Saffir-Simpson category did Hurricane Milton make landfall near Siesta Key, Florida on 9 October 2024? | Category 3 / Category 3 | Category 3 | exact_alias_match | none recorded |
| 22 | 2024-10-22 | What new capability, letting the model control a computer&#x27;s screen, cursor and keyboard, did Anthropic announce in public beta alongside the upgraded Claude 3.5 Sonnet on 22 October 2024? | Computer use / Computer use | Computer use | exact_alias_match | answer_inferable_from_definition |
| 23 | 2024-11 | Who won the 2024 United States presidential election? | Donald Trump / trump | Donald Trump | exact_alias_match | none recorded |
| 24 | 2024-11 | Who won the November 2024 boxing match between Jake Paul and Mike Tyson? | Jake Paul / paul | Jake Paul | exact_alias_match | named_candidate_choice |
| 25 | 2024-11-05 | Which party won control of the US Senate in the 5 November 2024 elections, and with how many seats? | Republicans, 53 seats / Republicans, 53 seats | Republican Party, 53 seats | manual_review | compound_exact_alias_is_narrow |
| 26 | 2024-12 | Which Syrian president was overthrown when rebels took Damascus in December 2024? | Bashar al-Assad / assad | Bashar al-Assad | exact_alias_match | none recorded |
| 27 | 2024-12 | Which country&#x27;s president declared a short-lived martial law in December 2024? | South Korea / south korea | South Korea | exact_alias_match | none recorded |
| 28 | 2024-12 | Which former US president died on December 29, 2024? | Jimmy Carter / carter | Jimmy Carter | exact_alias_match | none recorded |
| 29 | 2024-12 | The CEO of which health-insurance company was fatally shot in Manhattan in December 2024? | UnitedHealthcare / unitedhealth | UnitedHealthcare | exact_alias_match | none recorded |
| 30 | 2024-12 | Who won the 2024 World Chess Championship? | Gukesh Dommaraju / gukesh | Ding Liren | manual_review | none recorded |
| 31 | 2024-12 | Which Paris cathedral reopened in December 2024 after restoration from its 2019 fire? | Notre-Dame / notre | Notre-Dame de Paris | manual_review | none recorded |
| 32 | 2024-12-03 | Which South Korean president declared emergency martial law on 3 December 2024, rescinded within about six hours? | Yoon Suk Yeol / Yoon Suk Yeol | Yoon Suk Yeol | exact_alias_match | none recorded |
| 33 | 2024-12-08 | To which country did Bashar al-Assad flee when his regime fell to rebel forces on 8 December 2024? | Russia / Russia | Russia | exact_alias_match | none recorded |
| 34 | 2025-01 | Which Chinese AI company&#x27;s R1 reasoning model triggered a US tech-stock selloff in January 2025? | DeepSeek / deepseek | DeepSeek | exact_alias_match | none recorded |
| 35 | 2025-01 | Which US city was devastated by the Palisades and Eaton wildfires in January 2025? | Los Angeles / los angeles | Los Angeles | exact_alias_match | none recorded |
| 36 | 2025-01 | Who was inaugurated as the 47th president of the United States in January 2025? | Donald Trump / trump | Donald Trump | exact_alias_match | earlier_election_result_predicts_identity |
| 37 | 2025-01-07 | What was the name of the largest of the wildfires that broke out in Los Angeles County on January 7, 2025? | The Palisades Fire / The Palisades Fire | Palisades Fire | manual_review | cross_probe_context_cue |
| 38 | 2025-01-29 | How many people died in the midair collision between an American Airlines regional jet and a US Army Black Hawk helicopter over the Potomac River near Washington DC on January 29, 2025? | 67 / 67 | 67 people | manual_review | none recorded |
| 39 | 2025-02 | Which team won Super Bowl LIX in February 2025? | Philadelphia Eagles / eagles, philadelphia | Philadelphia Eagles | exact_alias_match | none recorded |
| 40 | 2025-02 | Which party bloc won Germany&#x27;s February 2025 federal election? | CDU/CSU / cdu, union | CDU/CSU (Union) | manual_review | none recorded |
| 41 | 2025-02 | Which Oscar-winning actor was found dead along with his wife at their New Mexico home in February 2025? | Gene Hackman / hackman | Gene Hackman | exact_alias_match | none recorded |
| 42 | 2025-02-24 | What hybrid reasoning model did Anthropic release on February 24, 2025, alongside the launch of Claude Code? | Claude 3.7 Sonnet / Claude 3.7 Sonnet | Claude 3.7 Sonnet | exact_alias_match | none recorded |
| 43 | 2025-02-27 | What model did OpenAI release as a research preview on February 27, 2025, described as its largest and last non-chain-of-thought model? | GPT-4.5 / GPT-4.5 | GPT-4.5 | exact_alias_match | release_does_not_establish_first_public_model_name |
| 44 | 2025-03 | Which film won Best Picture at the March 2025 Academy Awards? | Anora / anora | Anora | exact_alias_match | none recorded |
| 45 | 2025-03 | Who became Canada&#x27;s prime minister in March 2025? | Mark Carney / carney | Mark Carney | exact_alias_match | none recorded |
| 46 | 2025-03 | The editor-in-chief of which magazine was accidentally added to a Signal group chat of US officials discussing Yemen strikes in March 2025? | The Atlantic / atlantic | The Atlantic | exact_alias_match | none recorded |
| 47 | 2025-03 | Which country was struck by a devastating magnitude-7.7 earthquake in March 2025? | Myanmar / myanmar, burma | Myanmar | exact_alias_match | cross_probe_context_cue |
| 48 | 2025-03-25 | What AI model did Google announce on March 25, 2025, which debuted at #1 on LMArena? | Gemini 2.5 Pro / Gemini 2.5 Pro, Gemini 2.5 Pro (Experimental) | Gemini 2.5 Pro | exact_alias_match | none recorded |
| 49 | 2025-03-28 | What was the magnitude of the earthquake that struck near Mandalay, Myanmar on March 28, 2025? | 7.7 / 7.7 | 7.7 magnitude | manual_review | cross_probe_context_cue |
| 50 | 2025-04 | Which pope died in April 2025? | Pope Francis / francis | Pope Francis | exact_alias_match | none recorded |
| 51 | 2025-04 | What name did Donald Trump give to his sweeping tariff announcement of April 2, 2025? | Liberation Day / liberation | Liberation Day | exact_alias_match | name_announced_before_tariff_event |
| 52 | 2025-04 | Which party won Canada&#x27;s April 2025 federal election? | Liberal Party / liberal | Liberal Party | exact_alias_match | none recorded |
| 53 | 2025-05 | What papal name did Cardinal Robert Prevost take when elected pope in May 2025? | Leo XIV / leo | Leo XIV | exact_alias_match | ordinal_omitted_in_accepted_alias |
| 54 | 2025-05 | Which club won the 2024-25 UEFA Champions League final in May 2025? | Paris Saint-Germain / paris, psg | Paris Saint-Germain | exact_alias_match | city_only_club_alias |
| 55 | 2025-05 | Which party won Australia&#x27;s May 2025 federal election? | Labor / labor, albanese | Australian Labor Party | manual_review | person_alias_for_party_question |
| 56 | 2025-05 | Which two nuclear-armed neighbors fought a four-day military conflict in May 2025 following the Pahalgam attack? | India and Pakistan / pakistan | India and Pakistan | exact_alias_match | cross_probe_context_cue, one_country_alias_for_two_country_question |
| 57 | 2025-05-03 | Which horse won the 151st Kentucky Derby on May 3, 2025? | Sovereignty / Sovereignty | Sovereignty | exact_alias_match | none recorded |
| 58 | 2025-05-07 | What was the codename of India&#x27;s May 7, 2025 missile strikes on Pakistan, launched in response to the April 22 Pahalgam attack? | Operation Sindoor / Operation Sindoor | Operation Sindoor | exact_alias_match | none recorded |
| 59 | 2025-05-08 | Who was elected pope on May 8, 2025, becoming the first American pontiff, and what name did he take? | Cardinal Robert Francis Prevost, who took the name Leo XIV / Cardinal Robert Francis Prevost, who took the name Leo XIV | Pope Leo XIV (Robert Francis Prevost) | manual_review | cross_probe_context_cue, compound_exact_alias_is_narrow |
| 60 | 2025-06 | Which Iranian nuclear site, buried deep inside a mountain, was the primary target of the June 2025 US B-2 bunker-buster strikes? | Fordow / fordow, fordo | Fordow Fuel Enrichment Plant | manual_review | primary_target_is_editorial_characterization, cross_probe_context_cue |
| 61 | 2025-06 | In which Indian city did an Air India Boeing 787 crash shortly after takeoff in June 2025? | Ahmedabad / ahmedabad | Ahmedabad | exact_alias_match | none recorded |
| 62 | 2025-06 | Which team won the 2025 NBA Finals? | Oklahoma City Thunder / thunder, oklahoma | Oklahoma City Thunder | exact_alias_match | none recorded |
| 63 | 2025-06-13 | What was the codename of Israel&#x27;s large-scale air campaign against Iran&#x27;s nuclear program and military leadership launched June 13, 2025? | Operation Rising Lion / Operation Rising Lion | Operation Rising Lion | exact_alias_match | none recorded |
| 64 | 2025-06-22 | What was the codename of the US B-2 bomber strike on Iran&#x27;s Fordow, Natanz, and Isfahan nuclear sites on the night of June 21-22, 2025? | Operation Midnight Hammer / Operation Midnight Hammer | Operation Midnight Hammer | exact_alias_match | none recorded |
| 65 | 2025-07 | Who won the 2025 Wimbledon men&#x27;s singles title? | Jannik Sinner / sinner | Jannik Sinner | exact_alias_match | none recorded |
| 66 | 2025-07 | Which heavy-metal singer died in July 2025, weeks after his farewell concert in Birmingham? | Ozzy Osbourne / osbourne, ozzy | Ozzy Osbourne | exact_alias_match | none recorded |
| 67 | 2025-07 | Which club won the expanded FIFA Club World Cup final in July 2025? | Chelsea / chelsea | Chelsea FC | manual_review | none recorded |
| 68 | 2025-07 | Which professional-wrestling icon died in July 2025 at age 71? | Hulk Hogan / hogan | Hulk Hogan | exact_alias_match | none recorded |
| 69 | 2025-07-27 | Which national team won the UEFA Women&#x27;s Euro 2025 final on July 27, 2025, beating Spain on penalties in Basel? | England / England | England | exact_alias_match | none recorded |
| 70 | 2025-08 | What flagship model did OpenAI release in August 2025? | GPT-5 / gpt-5, gpt5 | GPT-5 | exact_alias_match | none recorded |
| 71 | 2025-08 | In which US state did Donald Trump and Vladimir Putin hold their August 2025 summit? | Alaska / alaska | Alaska | exact_alias_match | none recorded |
| 72 | 2025-09 | Which conservative activist was assassinated at Utah Valley University in September 2025? | Charlie Kirk / kirk | Charlie Kirk | exact_alias_match | cross_probe_context_cue |
| 73 | 2025-09 | Gen-Z-led protests in September 2025 forced the prime minister of which Himalayan country to resign? | Nepal / nepal | Nepal | exact_alias_match | none recorded |
| 74 | 2025-09 | Who was appointed France&#x27;s prime minister in September 2025? | Sébastien Lecornu / lecornu | Sébastien Lecornu | exact_alias_match | none recorded |
| 75 | 2025-09 | Who won the 2025 US Open men&#x27;s singles title? | Carlos Alcaraz / alcaraz | Carlos Alcaraz | exact_alias_match | none recorded |
| 76 | 2025-09-10 | Conservative activist Charlie Kirk was assassinated at Utah Valley University on September 10, 2025. What is the name of the man charged with the shooting? | Tyler Robinson / Tyler Robinson | Tyler Robinson | exact_alias_match | shooting_date_precedes_requested_suspect_fact |
| 77 | 2025-09-22 | Which PSG forward won the men&#x27;s 2025 Ballon d&#x27;Or, awarded on September 22, 2025? | Ousmane Dembélé / Ousmane Dembélé | Ousmane Dembélé | exact_alias_match | diacritic_sensitive_exact_alias |
| 78 | 2025-09-27 | Which national team won the 2025 Women&#x27;s Rugby World Cup final at Twickenham on September 27, 2025, beating Canada 33-13? | England / England | England | exact_alias_match | none recorded |
| 79 | 2025-09-29 | Which AI model did Anthropic release on September 29, 2025, billing it as the best coding model in the world alongside Claude Code 2.0? | Claude Sonnet 4.5 / Claude Sonnet 4.5 | Claude Sonnet 4.5 | exact_alias_match | none recorded |
| 80 | 2025-10 | Which Paris museum had crown jewels stolen in a daylight heist in October 2025? | Louvre / louvre | The Louvre Museum | manual_review | none recorded |
| 81 | 2025-10 | Who won the 2025 Nobel Peace Prize? | María Corina Machado / machado | UNKNOWN | explicit_unknown | none recorded |
| 82 | 2025-10 | Who became Japan&#x27;s first female prime minister in October 2025? | Sanae Takaichi / takaichi | Sanae Takaichi | exact_alias_match | none recorded |
| 83 | 2025-10 | Under whose 20-point peace plan did Israel and Hamas agree an October 2025 ceasefire and hostage release? | Donald Trump / trump | Donald Trump | exact_alias_match | none recorded |
| 84 | 2025-10-13 | On October 13, 2025, under the Gaza ceasefire deal, Hamas released all remaining living Israeli hostages. How many living hostages were released that day? | 20 / 20 | 20 | exact_alias_match | none recorded |
| 85 | 2025-10-28 | Hurricane Melissa made landfall in Jamaica on October 28, 2025 as the strongest hurricane ever recorded to hit the island. What Saffir-Simpson category was it at landfall? | Category 5 / Category 5 | Category 5 | exact_alias_match | none recorded |
| 86 | 2025-11 | Who won the November 2025 New York City mayoral election? | Zohran Mamdani / mamdani | Zohran Mamdani | exact_alias_match | none recorded |
| 87 | 2025-11 | Which team won the 2025 World Series? | Los Angeles Dodgers / dodgers | UNKNOWN | explicit_unknown | none recorded |
| 88 | 2025-11 | Which former US vice president died in November 2025? | Dick Cheney / cheney | Dick Cheney | exact_alias_match | none recorded |
| 89 | 2025-11 | Which Brazilian city hosted the COP30 climate summit in November 2025? | Belém / belem, belém | Belém | exact_alias_match | host_known_before_benchmark_date_range |
| 90 | 2025-11 | What was Anthropic&#x27;s flagship Claude model release of November 2025 called? | Claude Opus 4.5 / opus 4.5, 4.5 | Claude Opus 4.5 | exact_alias_match | none recorded |
| 91 | 2025-11-12 | The longest US federal government shutdown in history ended on November 12, 2025 when President Trump signed a funding bill. How many days did it last? | 43 / 43 | 43 days | manual_review | none recorded |
| 92 | 2025-11-18 | Which flagship AI model did Google release on November 18, 2025, topping the LMArena leaderboard and launching alongside the Antigravity IDE? | Gemini 3 / Gemini 3 | Gemini 3 | exact_alias_match | none recorded |
| 93 | 2025-12 | Who won Chile&#x27;s December 2025 presidential runoff? | José Antonio Kast / kast | José Antonio Kast | exact_alias_match | none recorded |
| 94 | 2026-02 | Who won Super Bowl LX in February 2026? | Seattle Seahawks / seahawks, seattle | Seattle Seahawks | exact_alias_match | none recorded |
| 95 | 2026-02 | Which country topped the medal table at the 2026 Winter Olympics in Milan-Cortina? | Norway / norway | UNKNOWN | explicit_unknown | none recorded |
| 96 | 2026-02 | Which country won the gold medal in men&#x27;s ice hockey at the 2026 Winter Olympics? | United States / united states, usa, american | UNKNOWN | explicit_unknown | none recorded |
| 97 | 2026-02 | Who won Album of the Year at the Grammy Awards held in February 2026? | Bad Bunny (Debí Tirar Más Fotos) / bad bunny, debi tirar, debí tirar | UNKNOWN | explicit_unknown | none recorded |
| 98 | 2026-02 | Who won the men&#x27;s singles title at the 2026 Australian Open? | Carlos Alcaraz / alcaraz | UNKNOWN | explicit_unknown | none recorded |
| 99 | 2026-02 | Who won Costa Rica&#x27;s presidential election held on February 1, 2026? | Laura Fernández / fernandez, fernández | UNKNOWN | explicit_unknown | none recorded |
| 100 | 2026-02 | Which country&#x27;s supreme leader was assassinated in airstrikes on his capital on February 28, 2026? | Iran (Ali Khamenei) / iran, khamenei | Iran | exact_alias_match | person_alias_for_country_question |
| 101 | 2026-02-15 | Which Oscar-winning actor, star of The Godfather and Apocalypse Now, died on February 15, 2026 at age 95? | Robert Duvall / Robert Duvall | Robert Duvall | exact_alias_match | publication_after_event_day |
| 102 | 2026-03 | Which film won Best Picture at the 98th Academy Awards in March 2026? | One Battle After Another / one battle after another | One Battle After Another | exact_alias_match | none recorded |
| 103 | 2026-03 | Which country won the 2026 Six Nations rugby championship? | France / france | UNKNOWN | explicit_unknown | none recorded |
| 104 | 2026-03 | Which version of its flagship GPT model series did OpenAI release on March 5, 2026? | GPT-5.4 / 5.4 | GPT-5.4 | exact_alias_match | none recorded |
| 105 | 2026-03 | Who was elected Iran&#x27;s third supreme leader in March 2026? | Mojtaba Khamenei / mojtaba | UNKNOWN | explicit_unknown | none recorded |
| 106 | 2026-03-15 | Who won Best Actor at the 98th Academy Awards on March 15, 2026? | Michael B. Jordan / Michael B. Jordan, Michael B. Jordan (for Sinners) | Michael B. Jordan | exact_alias_match | none recorded |
| 107 | 2026-04 | Who became Prime Minister of Hungary following the April 2026 parliamentary election? | Péter Magyar / magyar | UNKNOWN | explicit_unknown | election_month_is_not_taking_office_month |
| 108 | 2026-04 | Which company released the AI model GPT-5.5 in April 2026? | OpenAI / openai | UNKNOWN | explicit_unknown | brand_to_company_inference |
| 109 | 2026-04-06 | Which team won the 2026 NCAA men&#x27;s basketball national championship on April 6, 2026? | Michigan Wolverines / Michigan Wolverines, Michigan Wolverines (beat UConn 69-63) | UNKNOWN | explicit_unknown | none recorded |
| 110 | 2026-05 | Which horse won the 2026 Kentucky Derby? | Golden Tempo / golden tempo | UNKNOWN | explicit_unknown | none recorded |
| 111 | 2026-05 | Which country won the Eurovision Song Contest 2026? | Bulgaria / bulgaria | UNKNOWN | explicit_unknown | none recorded |
| 112 | 2026-05 | Which film won the Palme d&#x27;Or at the 2026 Cannes Film Festival? | Fjord / fjord, mungiu | UNKNOWN | explicit_unknown | director_alias_for_film_question |
| 113 | 2026-05 | Which club won the 2026 UEFA Champions League final in Budapest? | Paris Saint-Germain / paris, psg | UNKNOWN | explicit_unknown | city_only_club_alias |
| 114 | 2026-06 | Who won the men&#x27;s singles title at the 2026 French Open? | Alexander Zverev / zverev | UNKNOWN | explicit_unknown | none recorded |
| 115 | 2026-06 | Which team won the 2026 NBA Finals? | New York Knicks / knicks | UNKNOWN | explicit_unknown | none recorded |
| 116 | 2026-06 | Which team won the 2026 Stanley Cup Final? | Carolina Hurricanes / hurricanes, carolina | UNKNOWN | explicit_unknown | none recorded |
| 117 | 2026-06 | Who won Colombia&#x27;s presidential runoff election on June 21, 2026? | Abelardo de la Espriella / espriella, abelardo | UNKNOWN | explicit_unknown | original_source_is_preliminary_result |
| 118 | 2026-06-06 | Who won the 2026 French Open women&#x27;s singles title on June 6, 2026? | Mirra Andreeva / Mirra Andreeva, Mirra Andreeva (beat Maja Chwalinska) | UNKNOWN | explicit_unknown | none recorded |
| 119 | 2026-06-09 | Which flagship AI model did Anthropic release on June 9, 2026, alongside the restricted Mythos 5? | Claude Fable 5 / Claude Fable 5 | UNKNOWN | explicit_unknown | none recorded |
| 120 | 2026-06-24 | Which country was struck by twin magnitude 7.2 and 7.5 earthquakes on June 24, 2026, its strongest since 1900, killing over 2,000 people? | Venezuela / Venezuela | UNKNOWN | explicit_unknown | time_varying_casualty_premise |
| 121 | 2026-07 | Which country won the 2026 FIFA World Cup final on July 19, 2026? | Spain / spain | UNKNOWN | explicit_unknown | none recorded |
| 122 | 2026-07 | Who won the men&#x27;s singles title at Wimbledon 2026? | Jannik Sinner / sinner | UNKNOWN | explicit_unknown | none recorded |
| 123 | 2026-07 | Who won the women&#x27;s singles title at Wimbledon 2026? | Linda Nosková / noskova, nosková | UNKNOWN | explicit_unknown | none recorded |
| 124 | 2026-07 | Who won the 2026 Tour de France general classification? | Tadej Pogačar / pogacar, pogačar | UNKNOWN | explicit_unknown | none recorded |
| 125 | 2026-07 | Who won the 2026 Open Championship (British Open) at Royal Birkdale? | Ryan Fox / ryan fox | UNKNOWN | explicit_unknown | none recorded |
| 126 | 2026-07 | Which company released the AI model Claude Opus 5 on July 24, 2026? | Anthropic / anthropic | Anthropic | exact_alias_match | brand_to_company_inference |
| 127 | 2026-07 | Which version of its flagship GPT model series did OpenAI release on July 9, 2026? | GPT-5.6 / 5.6 | UNKNOWN | explicit_unknown | limited_preview_precedes_public_release |
| 128 | 2026-08 | Which country topped the medal table at the 2026 Commonwealth Games in Glasgow? | Australia / australia | UNKNOWN | explicit_unknown | none recorded |
| 129 | 2026-08 | Which actress, star of Heroes and Nashville, died on August 16, 2026, at age 36? | Hayden Panettiere / panettiere | UNKNOWN | explicit_unknown | none recorded |

</details>

<details><summary>gpt-5.6-luna · xhigh · all 129 probes</summary>

| # | Recorded event date | Question | Original answer / aliases | Full response | Original-key grade | Source flags |
| ---: | --- | --- | --- | --- | --- | --- |
| 1 | 2024-07 | Who became the Democratic Party&#x27;s presidential nominee after Joe Biden withdrew from the 2024 race? | Kamala Harris / harris | Kamala Harris | exact_alias_match | nominee_stage_date_mismatch |
| 2 | 2024-07 | In which US state did the July 2024 assassination attempt on Donald Trump take place? | Pennsylvania / pennsylvania | Pennsylvania | exact_alias_match | cross_probe_context_cue |
| 3 | 2024-07 | Which party won the July 2024 United Kingdom general election? | Labour / labour, labor | Labour Party | manual_review | cross_probe_context_cue |
| 4 | 2024-07 | A faulty update from which cybersecurity company caused a global Windows outage in July 2024? | CrowdStrike / crowdstrike | CrowdStrike | exact_alias_match | none recorded |
| 5 | 2024-07 | Which national team won UEFA Euro 2024? | Spain / spain | Spain | exact_alias_match | none recorded |
| 6 | 2024-07 | Which national team won the 2024 Copa América? | Argentina / argentina | Argentina | exact_alias_match | none recorded |
| 7 | 2024-07-05 | Who became UK Prime Minister on 5 July 2024 after Labour&#x27;s landslide win in the July 4 general election? | Keir Starmer / Keir Starmer | Keir Starmer | exact_alias_match | none recorded |
| 8 | 2024-07-13 | In which Pennsylvania town was Donald Trump wounded in an assassination attempt at a campaign rally on 13 July 2024? | Butler / Butler | Butler, Pennsylvania | manual_review | near_city_not_within_city |
| 9 | 2024-08 | Which country&#x27;s prime minister, Sheikh Hasina, resigned and fled the country in August 2024? | Bangladesh / bangladesh | Bangladesh | exact_alias_match | none recorded |
| 10 | 2024-08 | Which country won the most total medals at the Paris 2024 Olympics? | United States / united states, usa, u.s. | United States | exact_alias_match | none recorded |
| 11 | 2024-08-04 | Who won the men&#x27;s 100m final at the Paris 2024 Olympics on 4 August 2024, in a photo finish over Kishane Thompson? | Noah Lyles / Noah Lyles | Noah Lyles | exact_alias_match | none recorded |
| 12 | 2024-08-11 | How many gold medals did both the United States and China each finish with atop the Paris 2024 Olympics medal table (closing 11 August 2024)? | 40 / 40 | 40 gold medals | manual_review | none recorded |
| 13 | 2024-08-18 | Which French film legend, star of &#x27;Le Samouraï&#x27; and &#x27;Purple Noon&#x27;, died on 18 August 2024 at age 88? | Alain Delon / Alain Delon | Alain Delon | exact_alias_match | none recorded |
| 14 | 2024-09 | In which country did thousands of pagers used by Hezbollah members explode in September 2024? | Lebanon / lebanon | Lebanon | exact_alias_match | geographic_scope_not_exclusive |
| 15 | 2024-09 | Which Hezbollah secretary-general was killed in an Israeli airstrike in September 2024? | Hassan Nasrallah / nasrallah | Hassan Nasrallah | exact_alias_match | none recorded |
| 16 | 2024-09-12 | What was the name of the reasoning model family OpenAI released in preview on 12 September 2024? | o1 / o1, o1 (o1-preview) | o1 | exact_alias_match | none recorded |
| 17 | 2024-09-26 | At what Saffir-Simpson category did Hurricane Helene make landfall in Florida&#x27;s Big Bend on 26 September 2024? | Category 4 / Category 4 | Category 4 | exact_alias_match | none recorded |
| 18 | 2024-10 | Which organization won the 2024 Nobel Peace Prize? | Nihon Hidankyo / hidankyo | Nihon Hidankyo | exact_alias_match | none recorded |
| 19 | 2024-10 | Which team won the 2024 World Series? | Los Angeles Dodgers / dodgers | Los Angeles Dodgers | exact_alias_match | none recorded |
| 20 | 2024-10-08 | Which two scientists shared the 2024 Nobel Prize in Physics (announced 8 October 2024) for foundational work on machine learning with artificial neural networks? | John Hopfield and Geoffrey Hinton / John Hopfield and Geoffrey Hinton | John J. Hopfield and Geoffrey E. Hinton | manual_review | compound_exact_alias_is_narrow |
| 21 | 2024-10-09 | At what Saffir-Simpson category did Hurricane Milton make landfall near Siesta Key, Florida on 9 October 2024? | Category 3 / Category 3 | Category 3 | exact_alias_match | none recorded |
| 22 | 2024-10-22 | What new capability, letting the model control a computer&#x27;s screen, cursor and keyboard, did Anthropic announce in public beta alongside the upgraded Claude 3.5 Sonnet on 22 October 2024? | Computer use / Computer use | Computer use | exact_alias_match | answer_inferable_from_definition |
| 23 | 2024-11 | Who won the 2024 United States presidential election? | Donald Trump / trump | Donald Trump | exact_alias_match | none recorded |
| 24 | 2024-11 | Who won the November 2024 boxing match between Jake Paul and Mike Tyson? | Jake Paul / paul | Jake Paul | exact_alias_match | named_candidate_choice |
| 25 | 2024-11-05 | Which party won control of the US Senate in the 5 November 2024 elections, and with how many seats? | Republicans, 53 seats / Republicans, 53 seats | Republican Party, with 53 seats | manual_review | compound_exact_alias_is_narrow |
| 26 | 2024-12 | Which Syrian president was overthrown when rebels took Damascus in December 2024? | Bashar al-Assad / assad | Bashar al-Assad | exact_alias_match | none recorded |
| 27 | 2024-12 | Which country&#x27;s president declared a short-lived martial law in December 2024? | South Korea / south korea | South Korea | exact_alias_match | none recorded |
| 28 | 2024-12 | Which former US president died on December 29, 2024? | Jimmy Carter / carter | Jimmy Carter | exact_alias_match | none recorded |
| 29 | 2024-12 | The CEO of which health-insurance company was fatally shot in Manhattan in December 2024? | UnitedHealthcare / unitedhealth | UnitedHealthcare | exact_alias_match | none recorded |
| 30 | 2024-12 | Who won the 2024 World Chess Championship? | Gukesh Dommaraju / gukesh | Gukesh Dommaraju | exact_alias_match | none recorded |
| 31 | 2024-12 | Which Paris cathedral reopened in December 2024 after restoration from its 2019 fire? | Notre-Dame / notre | Notre-Dame de Paris | manual_review | none recorded |
| 32 | 2024-12-03 | Which South Korean president declared emergency martial law on 3 December 2024, rescinded within about six hours? | Yoon Suk Yeol / Yoon Suk Yeol | Yoon Suk Yeol | exact_alias_match | none recorded |
| 33 | 2024-12-08 | To which country did Bashar al-Assad flee when his regime fell to rebel forces on 8 December 2024? | Russia / Russia | Russia | exact_alias_match | none recorded |
| 34 | 2025-01 | Which Chinese AI company&#x27;s R1 reasoning model triggered a US tech-stock selloff in January 2025? | DeepSeek / deepseek | DeepSeek | exact_alias_match | none recorded |
| 35 | 2025-01 | Which US city was devastated by the Palisades and Eaton wildfires in January 2025? | Los Angeles / los angeles | Los Angeles | exact_alias_match | none recorded |
| 36 | 2025-01 | Who was inaugurated as the 47th president of the United States in January 2025? | Donald Trump / trump | Donald Trump | exact_alias_match | earlier_election_result_predicts_identity |
| 37 | 2025-01-07 | What was the name of the largest of the wildfires that broke out in Los Angeles County on January 7, 2025? | The Palisades Fire / The Palisades Fire | Palisades Fire | manual_review | cross_probe_context_cue |
| 38 | 2025-01-29 | How many people died in the midair collision between an American Airlines regional jet and a US Army Black Hawk helicopter over the Potomac River near Washington DC on January 29, 2025? | 67 / 67 | 67 people | manual_review | none recorded |
| 39 | 2025-02 | Which team won Super Bowl LIX in February 2025? | Philadelphia Eagles / eagles, philadelphia | Philadelphia Eagles | exact_alias_match | none recorded |
| 40 | 2025-02 | Which party bloc won Germany&#x27;s February 2025 federal election? | CDU/CSU / cdu, union | CDU/CSU coalition | manual_review | none recorded |
| 41 | 2025-02 | Which Oscar-winning actor was found dead along with his wife at their New Mexico home in February 2025? | Gene Hackman / hackman | Gene Hackman | exact_alias_match | none recorded |
| 42 | 2025-02-24 | What hybrid reasoning model did Anthropic release on February 24, 2025, alongside the launch of Claude Code? | Claude 3.7 Sonnet / Claude 3.7 Sonnet | Claude 3.7 Sonnet | exact_alias_match | none recorded |
| 43 | 2025-02-27 | What model did OpenAI release as a research preview on February 27, 2025, described as its largest and last non-chain-of-thought model? | GPT-4.5 / GPT-4.5 | GPT-4.5 | exact_alias_match | release_does_not_establish_first_public_model_name |
| 44 | 2025-03 | Which film won Best Picture at the March 2025 Academy Awards? | Anora / anora | Anora | exact_alias_match | none recorded |
| 45 | 2025-03 | Who became Canada&#x27;s prime minister in March 2025? | Mark Carney / carney | Mark Carney | exact_alias_match | none recorded |
| 46 | 2025-03 | The editor-in-chief of which magazine was accidentally added to a Signal group chat of US officials discussing Yemen strikes in March 2025? | The Atlantic / atlantic | The Atlantic | exact_alias_match | none recorded |
| 47 | 2025-03 | Which country was struck by a devastating magnitude-7.7 earthquake in March 2025? | Myanmar / myanmar, burma | Myanmar | exact_alias_match | cross_probe_context_cue |
| 48 | 2025-03-25 | What AI model did Google announce on March 25, 2025, which debuted at #1 on LMArena? | Gemini 2.5 Pro / Gemini 2.5 Pro, Gemini 2.5 Pro (Experimental) | Gemini 2.5 Pro | exact_alias_match | none recorded |
| 49 | 2025-03-28 | What was the magnitude of the earthquake that struck near Mandalay, Myanmar on March 28, 2025? | 7.7 / 7.7 | 7.7 magnitude | manual_review | cross_probe_context_cue |
| 50 | 2025-04 | Which pope died in April 2025? | Pope Francis / francis | Pope Francis | exact_alias_match | none recorded |
| 51 | 2025-04 | What name did Donald Trump give to his sweeping tariff announcement of April 2, 2025? | Liberation Day / liberation | Liberation Day | exact_alias_match | name_announced_before_tariff_event |
| 52 | 2025-04 | Which party won Canada&#x27;s April 2025 federal election? | Liberal Party / liberal | Liberal Party of Canada | manual_review | none recorded |
| 53 | 2025-05 | What papal name did Cardinal Robert Prevost take when elected pope in May 2025? | Leo XIV / leo | Leo XIV | exact_alias_match | ordinal_omitted_in_accepted_alias |
| 54 | 2025-05 | Which club won the 2024-25 UEFA Champions League final in May 2025? | Paris Saint-Germain / paris, psg | Paris Saint-Germain | exact_alias_match | city_only_club_alias |
| 55 | 2025-05 | Which party won Australia&#x27;s May 2025 federal election? | Labor / labor, albanese | Australian Labor Party | manual_review | person_alias_for_party_question |
| 56 | 2025-05 | Which two nuclear-armed neighbors fought a four-day military conflict in May 2025 following the Pahalgam attack? | India and Pakistan / pakistan | India and Pakistan | exact_alias_match | cross_probe_context_cue, one_country_alias_for_two_country_question |
| 57 | 2025-05-03 | Which horse won the 151st Kentucky Derby on May 3, 2025? | Sovereignty / Sovereignty | Sovereignty | exact_alias_match | none recorded |
| 58 | 2025-05-07 | What was the codename of India&#x27;s May 7, 2025 missile strikes on Pakistan, launched in response to the April 22 Pahalgam attack? | Operation Sindoor / Operation Sindoor | Operation Sindoor | exact_alias_match | none recorded |
| 59 | 2025-05-08 | Who was elected pope on May 8, 2025, becoming the first American pontiff, and what name did he take? | Cardinal Robert Francis Prevost, who took the name Leo XIV / Cardinal Robert Francis Prevost, who took the name Leo XIV | Robert Francis Prevost — Leo XIV | manual_review | cross_probe_context_cue, compound_exact_alias_is_narrow |
| 60 | 2025-06 | Which Iranian nuclear site, buried deep inside a mountain, was the primary target of the June 2025 US B-2 bunker-buster strikes? | Fordow / fordow, fordo | Fordow Fuel Enrichment Plant | manual_review | primary_target_is_editorial_characterization, cross_probe_context_cue |
| 61 | 2025-06 | In which Indian city did an Air India Boeing 787 crash shortly after takeoff in June 2025? | Ahmedabad / ahmedabad | Ahmedabad | exact_alias_match | none recorded |
| 62 | 2025-06 | Which team won the 2025 NBA Finals? | Oklahoma City Thunder / thunder, oklahoma | Oklahoma City Thunder | exact_alias_match | none recorded |
| 63 | 2025-06-13 | What was the codename of Israel&#x27;s large-scale air campaign against Iran&#x27;s nuclear program and military leadership launched June 13, 2025? | Operation Rising Lion / Operation Rising Lion | Operation Rising Lion | exact_alias_match | none recorded |
| 64 | 2025-06-22 | What was the codename of the US B-2 bomber strike on Iran&#x27;s Fordow, Natanz, and Isfahan nuclear sites on the night of June 21-22, 2025? | Operation Midnight Hammer / Operation Midnight Hammer | Operation Midnight Hammer | exact_alias_match | none recorded |
| 65 | 2025-07 | Who won the 2025 Wimbledon men&#x27;s singles title? | Jannik Sinner / sinner | Jannik Sinner | exact_alias_match | none recorded |
| 66 | 2025-07 | Which heavy-metal singer died in July 2025, weeks after his farewell concert in Birmingham? | Ozzy Osbourne / osbourne, ozzy | Ozzy Osbourne | exact_alias_match | none recorded |
| 67 | 2025-07 | Which club won the expanded FIFA Club World Cup final in July 2025? | Chelsea / chelsea | Chelsea | exact_alias_match | none recorded |
| 68 | 2025-07 | Which professional-wrestling icon died in July 2025 at age 71? | Hulk Hogan / hogan | Hulk Hogan | exact_alias_match | none recorded |
| 69 | 2025-07-27 | Which national team won the UEFA Women&#x27;s Euro 2025 final on July 27, 2025, beating Spain on penalties in Basel? | England / England | England | exact_alias_match | none recorded |
| 70 | 2025-08 | What flagship model did OpenAI release in August 2025? | GPT-5 / gpt-5, gpt5 | GPT-5 | exact_alias_match | none recorded |
| 71 | 2025-08 | In which US state did Donald Trump and Vladimir Putin hold their August 2025 summit? | Alaska / alaska | Alaska | exact_alias_match | none recorded |
| 72 | 2025-09 | Which conservative activist was assassinated at Utah Valley University in September 2025? | Charlie Kirk / kirk | Charlie Kirk | exact_alias_match | cross_probe_context_cue |
| 73 | 2025-09 | Gen-Z-led protests in September 2025 forced the prime minister of which Himalayan country to resign? | Nepal / nepal | Nepal | exact_alias_match | none recorded |
| 74 | 2025-09 | Who was appointed France&#x27;s prime minister in September 2025? | Sébastien Lecornu / lecornu | Sébastien Lecornu | exact_alias_match | none recorded |
| 75 | 2025-09 | Who won the 2025 US Open men&#x27;s singles title? | Carlos Alcaraz / alcaraz | Carlos Alcaraz | exact_alias_match | none recorded |
| 76 | 2025-09-10 | Conservative activist Charlie Kirk was assassinated at Utah Valley University on September 10, 2025. What is the name of the man charged with the shooting? | Tyler Robinson / Tyler Robinson | Tyler James Robinson | manual_review | shooting_date_precedes_requested_suspect_fact |
| 77 | 2025-09-22 | Which PSG forward won the men&#x27;s 2025 Ballon d&#x27;Or, awarded on September 22, 2025? | Ousmane Dembélé / Ousmane Dembélé | Ousmane Dembélé | exact_alias_match | diacritic_sensitive_exact_alias |
| 78 | 2025-09-27 | Which national team won the 2025 Women&#x27;s Rugby World Cup final at Twickenham on September 27, 2025, beating Canada 33-13? | England / England | England | exact_alias_match | none recorded |
| 79 | 2025-09-29 | Which AI model did Anthropic release on September 29, 2025, billing it as the best coding model in the world alongside Claude Code 2.0? | Claude Sonnet 4.5 / Claude Sonnet 4.5 | Claude Sonnet 4.5 | exact_alias_match | none recorded |
| 80 | 2025-10 | Which Paris museum had crown jewels stolen in a daylight heist in October 2025? | Louvre / louvre | The Louvre Museum | manual_review | none recorded |
| 81 | 2025-10 | Who won the 2025 Nobel Peace Prize? | María Corina Machado / machado | María Corina Machado | exact_alias_match | none recorded |
| 82 | 2025-10 | Who became Japan&#x27;s first female prime minister in October 2025? | Sanae Takaichi / takaichi | Sanae Takaichi | exact_alias_match | none recorded |
| 83 | 2025-10 | Under whose 20-point peace plan did Israel and Hamas agree an October 2025 ceasefire and hostage release? | Donald Trump / trump | Donald Trump’s 20-point peace plan | manual_review | none recorded |
| 84 | 2025-10-13 | On October 13, 2025, under the Gaza ceasefire deal, Hamas released all remaining living Israeli hostages. How many living hostages were released that day? | 20 / 20 | 20 living hostages | manual_review | none recorded |
| 85 | 2025-10-28 | Hurricane Melissa made landfall in Jamaica on October 28, 2025 as the strongest hurricane ever recorded to hit the island. What Saffir-Simpson category was it at landfall? | Category 5 / Category 5 | Category 5 | exact_alias_match | none recorded |
| 86 | 2025-11 | Who won the November 2025 New York City mayoral election? | Zohran Mamdani / mamdani | Zohran Mamdani | exact_alias_match | none recorded |
| 87 | 2025-11 | Which team won the 2025 World Series? | Los Angeles Dodgers / dodgers | UNKNOWN | explicit_unknown | none recorded |
| 88 | 2025-11 | Which former US vice president died in November 2025? | Dick Cheney / cheney | Dick Cheney | exact_alias_match | none recorded |
| 89 | 2025-11 | Which Brazilian city hosted the COP30 climate summit in November 2025? | Belém / belem, belém | Belém | exact_alias_match | host_known_before_benchmark_date_range |
| 90 | 2025-11 | What was Anthropic&#x27;s flagship Claude model release of November 2025 called? | Claude Opus 4.5 / opus 4.5, 4.5 | Claude Opus 4.5 | exact_alias_match | none recorded |
| 91 | 2025-11-12 | The longest US federal government shutdown in history ended on November 12, 2025 when President Trump signed a funding bill. How many days did it last? | 43 / 43 | 43 days | manual_review | none recorded |
| 92 | 2025-11-18 | Which flagship AI model did Google release on November 18, 2025, topping the LMArena leaderboard and launching alongside the Antigravity IDE? | Gemini 3 / Gemini 3 | Gemini 3 Pro | manual_review | none recorded |
| 93 | 2025-12 | Who won Chile&#x27;s December 2025 presidential runoff? | José Antonio Kast / kast | UNKNOWN | explicit_unknown | none recorded |
| 94 | 2026-02 | Who won Super Bowl LX in February 2026? | Seattle Seahawks / seahawks, seattle | Seattle Seahawks | exact_alias_match | none recorded |
| 95 | 2026-02 | Which country topped the medal table at the 2026 Winter Olympics in Milan-Cortina? | Norway / norway | Norway | exact_alias_match | none recorded |
| 96 | 2026-02 | Which country won the gold medal in men&#x27;s ice hockey at the 2026 Winter Olympics? | United States / united states, usa, american | UNKNOWN | explicit_unknown | none recorded |
| 97 | 2026-02 | Who won Album of the Year at the Grammy Awards held in February 2026? | Bad Bunny (Debí Tirar Más Fotos) / bad bunny, debi tirar, debí tirar | Bad Bunny | exact_alias_match | none recorded |
| 98 | 2026-02 | Who won the men&#x27;s singles title at the 2026 Australian Open? | Carlos Alcaraz / alcaraz | UNKNOWN | explicit_unknown | none recorded |
| 99 | 2026-02 | Who won Costa Rica&#x27;s presidential election held on February 1, 2026? | Laura Fernández / fernandez, fernández | Laura Fernández Delgado | manual_review | none recorded |
| 100 | 2026-02 | Which country&#x27;s supreme leader was assassinated in airstrikes on his capital on February 28, 2026? | Iran (Ali Khamenei) / iran, khamenei | Iran | exact_alias_match | person_alias_for_country_question |
| 101 | 2026-02-15 | Which Oscar-winning actor, star of The Godfather and Apocalypse Now, died on February 15, 2026 at age 95? | Robert Duvall / Robert Duvall | Robert Duvall | exact_alias_match | publication_after_event_day |
| 102 | 2026-03 | Which film won Best Picture at the 98th Academy Awards in March 2026? | One Battle After Another / one battle after another | UNKNOWN | explicit_unknown | none recorded |
| 103 | 2026-03 | Which country won the 2026 Six Nations rugby championship? | France / france | UNKNOWN | explicit_unknown | none recorded |
| 104 | 2026-03 | Which version of its flagship GPT model series did OpenAI release on March 5, 2026? | GPT-5.4 / 5.4 | UNKNOWN | explicit_unknown | none recorded |
| 105 | 2026-03 | Who was elected Iran&#x27;s third supreme leader in March 2026? | Mojtaba Khamenei / mojtaba | UNKNOWN | explicit_unknown | none recorded |
| 106 | 2026-03-15 | Who won Best Actor at the 98th Academy Awards on March 15, 2026? | Michael B. Jordan / Michael B. Jordan, Michael B. Jordan (for Sinners) | UNKNOWN | explicit_unknown | none recorded |
| 107 | 2026-04 | Who became Prime Minister of Hungary following the April 2026 parliamentary election? | Péter Magyar / magyar | UNKNOWN | explicit_unknown | election_month_is_not_taking_office_month |
| 108 | 2026-04 | Which company released the AI model GPT-5.5 in April 2026? | OpenAI / openai | UNKNOWN | explicit_unknown | brand_to_company_inference |
| 109 | 2026-04-06 | Which team won the 2026 NCAA men&#x27;s basketball national championship on April 6, 2026? | Michigan Wolverines / Michigan Wolverines, Michigan Wolverines (beat UConn 69-63) | UNKNOWN | explicit_unknown | none recorded |
| 110 | 2026-05 | Which horse won the 2026 Kentucky Derby? | Golden Tempo / golden tempo | UNKNOWN | explicit_unknown | none recorded |
| 111 | 2026-05 | Which country won the Eurovision Song Contest 2026? | Bulgaria / bulgaria | UNKNOWN | explicit_unknown | none recorded |
| 112 | 2026-05 | Which film won the Palme d&#x27;Or at the 2026 Cannes Film Festival? | Fjord / fjord, mungiu | UNKNOWN | explicit_unknown | director_alias_for_film_question |
| 113 | 2026-05 | Which club won the 2026 UEFA Champions League final in Budapest? | Paris Saint-Germain / paris, psg | UNKNOWN | explicit_unknown | city_only_club_alias |
| 114 | 2026-06 | Who won the men&#x27;s singles title at the 2026 French Open? | Alexander Zverev / zverev | UNKNOWN | explicit_unknown | none recorded |
| 115 | 2026-06 | Which team won the 2026 NBA Finals? | New York Knicks / knicks | UNKNOWN | explicit_unknown | none recorded |
| 116 | 2026-06 | Which team won the 2026 Stanley Cup Final? | Carolina Hurricanes / hurricanes, carolina | UNKNOWN | explicit_unknown | none recorded |
| 117 | 2026-06 | Who won Colombia&#x27;s presidential runoff election on June 21, 2026? | Abelardo de la Espriella / espriella, abelardo | UNKNOWN | explicit_unknown | original_source_is_preliminary_result |
| 118 | 2026-06-06 | Who won the 2026 French Open women&#x27;s singles title on June 6, 2026? | Mirra Andreeva / Mirra Andreeva, Mirra Andreeva (beat Maja Chwalinska) | UNKNOWN | explicit_unknown | none recorded |
| 119 | 2026-06-09 | Which flagship AI model did Anthropic release on June 9, 2026, alongside the restricted Mythos 5? | Claude Fable 5 / Claude Fable 5 | UNKNOWN | explicit_unknown | none recorded |
| 120 | 2026-06-24 | Which country was struck by twin magnitude 7.2 and 7.5 earthquakes on June 24, 2026, its strongest since 1900, killing over 2,000 people? | Venezuela / Venezuela | UNKNOWN | explicit_unknown | time_varying_casualty_premise |
| 121 | 2026-07 | Which country won the 2026 FIFA World Cup final on July 19, 2026? | Spain / spain | UNKNOWN | explicit_unknown | none recorded |
| 122 | 2026-07 | Who won the men&#x27;s singles title at Wimbledon 2026? | Jannik Sinner / sinner | UNKNOWN | explicit_unknown | none recorded |
| 123 | 2026-07 | Who won the women&#x27;s singles title at Wimbledon 2026? | Linda Nosková / noskova, nosková | UNKNOWN | explicit_unknown | none recorded |
| 124 | 2026-07 | Who won the 2026 Tour de France general classification? | Tadej Pogačar / pogacar, pogačar | UNKNOWN | explicit_unknown | none recorded |
| 125 | 2026-07 | Who won the 2026 Open Championship (British Open) at Royal Birkdale? | Ryan Fox / ryan fox | UNKNOWN | explicit_unknown | none recorded |
| 126 | 2026-07 | Which company released the AI model Claude Opus 5 on July 24, 2026? | Anthropic / anthropic | UNKNOWN | explicit_unknown | brand_to_company_inference |
| 127 | 2026-07 | Which version of its flagship GPT model series did OpenAI release on July 9, 2026? | GPT-5.6 / 5.6 | UNKNOWN | explicit_unknown | limited_preview_precedes_public_release |
| 128 | 2026-08 | Which country topped the medal table at the 2026 Commonwealth Games in Glasgow? | Australia / australia | UNKNOWN | explicit_unknown | none recorded |
| 129 | 2026-08 | Which actress, star of Heroes and Nashville, died on August 16, 2026, at age 36? | Hayden Panettiere / panettiere | Hayden Panettiere | exact_alias_match | none recorded |

</details>

<details><summary>gpt-5.6-luna · max · all 129 probes</summary>

| # | Recorded event date | Question | Original answer / aliases | Full response | Original-key grade | Source flags |
| ---: | --- | --- | --- | --- | --- | --- |
| 1 | 2024-07 | Who became the Democratic Party&#x27;s presidential nominee after Joe Biden withdrew from the 2024 race? | Kamala Harris / harris | Kamala Harris | exact_alias_match | nominee_stage_date_mismatch |
| 2 | 2024-07 | In which US state did the July 2024 assassination attempt on Donald Trump take place? | Pennsylvania / pennsylvania | Pennsylvania | exact_alias_match | cross_probe_context_cue |
| 3 | 2024-07 | Which party won the July 2024 United Kingdom general election? | Labour / labour, labor | Labour Party | manual_review | cross_probe_context_cue |
| 4 | 2024-07 | A faulty update from which cybersecurity company caused a global Windows outage in July 2024? | CrowdStrike / crowdstrike | CrowdStrike | exact_alias_match | none recorded |
| 5 | 2024-07 | Which national team won UEFA Euro 2024? | Spain / spain | Spain | exact_alias_match | none recorded |
| 6 | 2024-07 | Which national team won the 2024 Copa América? | Argentina / argentina | Argentina | exact_alias_match | none recorded |
| 7 | 2024-07-05 | Who became UK Prime Minister on 5 July 2024 after Labour&#x27;s landslide win in the July 4 general election? | Keir Starmer / Keir Starmer | Keir Starmer | exact_alias_match | none recorded |
| 8 | 2024-07-13 | In which Pennsylvania town was Donald Trump wounded in an assassination attempt at a campaign rally on 13 July 2024? | Butler / Butler | Butler, Pennsylvania | manual_review | near_city_not_within_city |
| 9 | 2024-08 | Which country&#x27;s prime minister, Sheikh Hasina, resigned and fled the country in August 2024? | Bangladesh / bangladesh | Bangladesh | exact_alias_match | none recorded |
| 10 | 2024-08 | Which country won the most total medals at the Paris 2024 Olympics? | United States / united states, usa, u.s. | United States | exact_alias_match | none recorded |
| 11 | 2024-08-04 | Who won the men&#x27;s 100m final at the Paris 2024 Olympics on 4 August 2024, in a photo finish over Kishane Thompson? | Noah Lyles / Noah Lyles | Noah Lyles | exact_alias_match | none recorded |
| 12 | 2024-08-11 | How many gold medals did both the United States and China each finish with atop the Paris 2024 Olympics medal table (closing 11 August 2024)? | 40 / 40 | 40 gold medals each | manual_review | none recorded |
| 13 | 2024-08-18 | Which French film legend, star of &#x27;Le Samouraï&#x27; and &#x27;Purple Noon&#x27;, died on 18 August 2024 at age 88? | Alain Delon / Alain Delon | Alain Delon | exact_alias_match | none recorded |
| 14 | 2024-09 | In which country did thousands of pagers used by Hezbollah members explode in September 2024? | Lebanon / lebanon | Lebanon | exact_alias_match | geographic_scope_not_exclusive |
| 15 | 2024-09 | Which Hezbollah secretary-general was killed in an Israeli airstrike in September 2024? | Hassan Nasrallah / nasrallah | Hassan Nasrallah | exact_alias_match | none recorded |
| 16 | 2024-09-12 | What was the name of the reasoning model family OpenAI released in preview on 12 September 2024? | o1 / o1, o1 (o1-preview) | OpenAI o1 | manual_review | none recorded |
| 17 | 2024-09-26 | At what Saffir-Simpson category did Hurricane Helene make landfall in Florida&#x27;s Big Bend on 26 September 2024? | Category 4 / Category 4 | Category 4 hurricane | manual_review | none recorded |
| 18 | 2024-10 | Which organization won the 2024 Nobel Peace Prize? | Nihon Hidankyo / hidankyo | Nihon Hidankyo | exact_alias_match | none recorded |
| 19 | 2024-10 | Which team won the 2024 World Series? | Los Angeles Dodgers / dodgers | Los Angeles Dodgers | exact_alias_match | none recorded |
| 20 | 2024-10-08 | Which two scientists shared the 2024 Nobel Prize in Physics (announced 8 October 2024) for foundational work on machine learning with artificial neural networks? | John Hopfield and Geoffrey Hinton / John Hopfield and Geoffrey Hinton | John Hopfield and Geoffrey Hinton | exact_alias_match | compound_exact_alias_is_narrow |
| 21 | 2024-10-09 | At what Saffir-Simpson category did Hurricane Milton make landfall near Siesta Key, Florida on 9 October 2024? | Category 3 / Category 3 | Category 3 hurricane | manual_review | none recorded |
| 22 | 2024-10-22 | What new capability, letting the model control a computer&#x27;s screen, cursor and keyboard, did Anthropic announce in public beta alongside the upgraded Claude 3.5 Sonnet on 22 October 2024? | Computer use / Computer use | Computer use | exact_alias_match | answer_inferable_from_definition |
| 23 | 2024-11 | Who won the 2024 United States presidential election? | Donald Trump / trump | Donald Trump | exact_alias_match | none recorded |
| 24 | 2024-11 | Who won the November 2024 boxing match between Jake Paul and Mike Tyson? | Jake Paul / paul | Jake Paul | exact_alias_match | named_candidate_choice |
| 25 | 2024-11-05 | Which party won control of the US Senate in the 5 November 2024 elections, and with how many seats? | Republicans, 53 seats / Republicans, 53 seats | Republican Party, 53 seats | manual_review | compound_exact_alias_is_narrow |
| 26 | 2024-12 | Which Syrian president was overthrown when rebels took Damascus in December 2024? | Bashar al-Assad / assad | Bashar al-Assad | exact_alias_match | none recorded |
| 27 | 2024-12 | Which country&#x27;s president declared a short-lived martial law in December 2024? | South Korea / south korea | South Korea | exact_alias_match | none recorded |
| 28 | 2024-12 | Which former US president died on December 29, 2024? | Jimmy Carter / carter | Jimmy Carter | exact_alias_match | none recorded |
| 29 | 2024-12 | The CEO of which health-insurance company was fatally shot in Manhattan in December 2024? | UnitedHealthcare / unitedhealth | UnitedHealthcare | exact_alias_match | none recorded |
| 30 | 2024-12 | Who won the 2024 World Chess Championship? | Gukesh Dommaraju / gukesh | Gukesh Dommaraju | exact_alias_match | none recorded |
| 31 | 2024-12 | Which Paris cathedral reopened in December 2024 after restoration from its 2019 fire? | Notre-Dame / notre | Notre-Dame de Paris | manual_review | none recorded |
| 32 | 2024-12-03 | Which South Korean president declared emergency martial law on 3 December 2024, rescinded within about six hours? | Yoon Suk Yeol / Yoon Suk Yeol | Yoon Suk Yeol | exact_alias_match | none recorded |
| 33 | 2024-12-08 | To which country did Bashar al-Assad flee when his regime fell to rebel forces on 8 December 2024? | Russia / Russia | Russia | exact_alias_match | none recorded |
| 34 | 2025-01 | Which Chinese AI company&#x27;s R1 reasoning model triggered a US tech-stock selloff in January 2025? | DeepSeek / deepseek | DeepSeek | exact_alias_match | none recorded |
| 35 | 2025-01 | Which US city was devastated by the Palisades and Eaton wildfires in January 2025? | Los Angeles / los angeles | Los Angeles | exact_alias_match | none recorded |
| 36 | 2025-01 | Who was inaugurated as the 47th president of the United States in January 2025? | Donald Trump / trump | Donald Trump | exact_alias_match | earlier_election_result_predicts_identity |
| 37 | 2025-01-07 | What was the name of the largest of the wildfires that broke out in Los Angeles County on January 7, 2025? | The Palisades Fire / The Palisades Fire | Palisades Fire | manual_review | cross_probe_context_cue |
| 38 | 2025-01-29 | How many people died in the midair collision between an American Airlines regional jet and a US Army Black Hawk helicopter over the Potomac River near Washington DC on January 29, 2025? | 67 / 67 | 67 people | manual_review | none recorded |
| 39 | 2025-02 | Which team won Super Bowl LIX in February 2025? | Philadelphia Eagles / eagles, philadelphia | Philadelphia Eagles | exact_alias_match | none recorded |
| 40 | 2025-02 | Which party bloc won Germany&#x27;s February 2025 federal election? | CDU/CSU / cdu, union | CDU/CSU (Union) | manual_review | none recorded |
| 41 | 2025-02 | Which Oscar-winning actor was found dead along with his wife at their New Mexico home in February 2025? | Gene Hackman / hackman | Gene Hackman | exact_alias_match | none recorded |
| 42 | 2025-02-24 | What hybrid reasoning model did Anthropic release on February 24, 2025, alongside the launch of Claude Code? | Claude 3.7 Sonnet / Claude 3.7 Sonnet | Claude 3.7 Sonnet | exact_alias_match | none recorded |
| 43 | 2025-02-27 | What model did OpenAI release as a research preview on February 27, 2025, described as its largest and last non-chain-of-thought model? | GPT-4.5 / GPT-4.5 | GPT-4.5 | exact_alias_match | release_does_not_establish_first_public_model_name |
| 44 | 2025-03 | Which film won Best Picture at the March 2025 Academy Awards? | Anora / anora | Anora | exact_alias_match | none recorded |
| 45 | 2025-03 | Who became Canada&#x27;s prime minister in March 2025? | Mark Carney / carney | Mark Carney | exact_alias_match | none recorded |
| 46 | 2025-03 | The editor-in-chief of which magazine was accidentally added to a Signal group chat of US officials discussing Yemen strikes in March 2025? | The Atlantic / atlantic | The Atlantic | exact_alias_match | none recorded |
| 47 | 2025-03 | Which country was struck by a devastating magnitude-7.7 earthquake in March 2025? | Myanmar / myanmar, burma | Myanmar | exact_alias_match | cross_probe_context_cue |
| 48 | 2025-03-25 | What AI model did Google announce on March 25, 2025, which debuted at #1 on LMArena? | Gemini 2.5 Pro / Gemini 2.5 Pro, Gemini 2.5 Pro (Experimental) | Gemini 2.5 Pro | exact_alias_match | none recorded |
| 49 | 2025-03-28 | What was the magnitude of the earthquake that struck near Mandalay, Myanmar on March 28, 2025? | 7.7 / 7.7 | 7.7 magnitude | manual_review | cross_probe_context_cue |
| 50 | 2025-04 | Which pope died in April 2025? | Pope Francis / francis | Pope Francis | exact_alias_match | none recorded |
| 51 | 2025-04 | What name did Donald Trump give to his sweeping tariff announcement of April 2, 2025? | Liberation Day / liberation | Liberation Day | exact_alias_match | name_announced_before_tariff_event |
| 52 | 2025-04 | Which party won Canada&#x27;s April 2025 federal election? | Liberal Party / liberal | Liberal Party of Canada | manual_review | none recorded |
| 53 | 2025-05 | What papal name did Cardinal Robert Prevost take when elected pope in May 2025? | Leo XIV / leo | Leo XIV | exact_alias_match | ordinal_omitted_in_accepted_alias |
| 54 | 2025-05 | Which club won the 2024-25 UEFA Champions League final in May 2025? | Paris Saint-Germain / paris, psg | Paris Saint-Germain | exact_alias_match | city_only_club_alias |
| 55 | 2025-05 | Which party won Australia&#x27;s May 2025 federal election? | Labor / labor, albanese | Australian Labor Party | manual_review | person_alias_for_party_question |
| 56 | 2025-05 | Which two nuclear-armed neighbors fought a four-day military conflict in May 2025 following the Pahalgam attack? | India and Pakistan / pakistan | India and Pakistan | exact_alias_match | cross_probe_context_cue, one_country_alias_for_two_country_question |
| 57 | 2025-05-03 | Which horse won the 151st Kentucky Derby on May 3, 2025? | Sovereignty / Sovereignty | Sovereignty | exact_alias_match | none recorded |
| 58 | 2025-05-07 | What was the codename of India&#x27;s May 7, 2025 missile strikes on Pakistan, launched in response to the April 22 Pahalgam attack? | Operation Sindoor / Operation Sindoor | Operation Sindoor | exact_alias_match | none recorded |
| 59 | 2025-05-08 | Who was elected pope on May 8, 2025, becoming the first American pontiff, and what name did he take? | Cardinal Robert Francis Prevost, who took the name Leo XIV / Cardinal Robert Francis Prevost, who took the name Leo XIV | Robert Francis Prevost, Pope Leo XIV | manual_review | cross_probe_context_cue, compound_exact_alias_is_narrow |
| 60 | 2025-06 | Which Iranian nuclear site, buried deep inside a mountain, was the primary target of the June 2025 US B-2 bunker-buster strikes? | Fordow / fordow, fordo | Fordow Fuel Enrichment Plant | manual_review | primary_target_is_editorial_characterization, cross_probe_context_cue |
| 61 | 2025-06 | In which Indian city did an Air India Boeing 787 crash shortly after takeoff in June 2025? | Ahmedabad / ahmedabad | Ahmedabad | exact_alias_match | none recorded |
| 62 | 2025-06 | Which team won the 2025 NBA Finals? | Oklahoma City Thunder / thunder, oklahoma | Oklahoma City Thunder | exact_alias_match | none recorded |
| 63 | 2025-06-13 | What was the codename of Israel&#x27;s large-scale air campaign against Iran&#x27;s nuclear program and military leadership launched June 13, 2025? | Operation Rising Lion / Operation Rising Lion | Operation Rising Lion | exact_alias_match | none recorded |
| 64 | 2025-06-22 | What was the codename of the US B-2 bomber strike on Iran&#x27;s Fordow, Natanz, and Isfahan nuclear sites on the night of June 21-22, 2025? | Operation Midnight Hammer / Operation Midnight Hammer | Operation Midnight Hammer | exact_alias_match | none recorded |
| 65 | 2025-07 | Who won the 2025 Wimbledon men&#x27;s singles title? | Jannik Sinner / sinner | Jannik Sinner | exact_alias_match | none recorded |
| 66 | 2025-07 | Which heavy-metal singer died in July 2025, weeks after his farewell concert in Birmingham? | Ozzy Osbourne / osbourne, ozzy | Ozzy Osbourne | exact_alias_match | none recorded |
| 67 | 2025-07 | Which club won the expanded FIFA Club World Cup final in July 2025? | Chelsea / chelsea | Chelsea | exact_alias_match | none recorded |
| 68 | 2025-07 | Which professional-wrestling icon died in July 2025 at age 71? | Hulk Hogan / hogan | Hulk Hogan | exact_alias_match | none recorded |
| 69 | 2025-07-27 | Which national team won the UEFA Women&#x27;s Euro 2025 final on July 27, 2025, beating Spain on penalties in Basel? | England / England | England | exact_alias_match | none recorded |
| 70 | 2025-08 | What flagship model did OpenAI release in August 2025? | GPT-5 / gpt-5, gpt5 | GPT-5 | exact_alias_match | none recorded |
| 71 | 2025-08 | In which US state did Donald Trump and Vladimir Putin hold their August 2025 summit? | Alaska / alaska | Alaska | exact_alias_match | none recorded |
| 72 | 2025-09 | Which conservative activist was assassinated at Utah Valley University in September 2025? | Charlie Kirk / kirk | Charlie Kirk | exact_alias_match | cross_probe_context_cue |
| 73 | 2025-09 | Gen-Z-led protests in September 2025 forced the prime minister of which Himalayan country to resign? | Nepal / nepal | Nepal | exact_alias_match | none recorded |
| 74 | 2025-09 | Who was appointed France&#x27;s prime minister in September 2025? | Sébastien Lecornu / lecornu | Sébastien Lecornu | exact_alias_match | none recorded |
| 75 | 2025-09 | Who won the 2025 US Open men&#x27;s singles title? | Carlos Alcaraz / alcaraz | Carlos Alcaraz | exact_alias_match | none recorded |
| 76 | 2025-09-10 | Conservative activist Charlie Kirk was assassinated at Utah Valley University on September 10, 2025. What is the name of the man charged with the shooting? | Tyler Robinson / Tyler Robinson | Tyler Robinson | exact_alias_match | shooting_date_precedes_requested_suspect_fact |
| 77 | 2025-09-22 | Which PSG forward won the men&#x27;s 2025 Ballon d&#x27;Or, awarded on September 22, 2025? | Ousmane Dembélé / Ousmane Dembélé | Ousmane Dembélé | exact_alias_match | diacritic_sensitive_exact_alias |
| 78 | 2025-09-27 | Which national team won the 2025 Women&#x27;s Rugby World Cup final at Twickenham on September 27, 2025, beating Canada 33-13? | England / England | England | exact_alias_match | none recorded |
| 79 | 2025-09-29 | Which AI model did Anthropic release on September 29, 2025, billing it as the best coding model in the world alongside Claude Code 2.0? | Claude Sonnet 4.5 / Claude Sonnet 4.5 | Claude Sonnet 4.5 | exact_alias_match | none recorded |
| 80 | 2025-10 | Which Paris museum had crown jewels stolen in a daylight heist in October 2025? | Louvre / louvre | Louvre Museum | manual_review | none recorded |
| 81 | 2025-10 | Who won the 2025 Nobel Peace Prize? | María Corina Machado / machado | UNKNOWN | explicit_unknown | none recorded |
| 82 | 2025-10 | Who became Japan&#x27;s first female prime minister in October 2025? | Sanae Takaichi / takaichi | Sanae Takaichi | exact_alias_match | none recorded |
| 83 | 2025-10 | Under whose 20-point peace plan did Israel and Hamas agree an October 2025 ceasefire and hostage release? | Donald Trump / trump | Donald Trump’s 20-point peace plan | manual_review | none recorded |
| 84 | 2025-10-13 | On October 13, 2025, under the Gaza ceasefire deal, Hamas released all remaining living Israeli hostages. How many living hostages were released that day? | 20 / 20 | 20 living hostages | manual_review | none recorded |
| 85 | 2025-10-28 | Hurricane Melissa made landfall in Jamaica on October 28, 2025 as the strongest hurricane ever recorded to hit the island. What Saffir-Simpson category was it at landfall? | Category 5 / Category 5 | Category 5 | exact_alias_match | none recorded |
| 86 | 2025-11 | Who won the November 2025 New York City mayoral election? | Zohran Mamdani / mamdani | Zohran Mamdani | exact_alias_match | none recorded |
| 87 | 2025-11 | Which team won the 2025 World Series? | Los Angeles Dodgers / dodgers | UNKNOWN | explicit_unknown | none recorded |
| 88 | 2025-11 | Which former US vice president died in November 2025? | Dick Cheney / cheney | Dick Cheney | exact_alias_match | none recorded |
| 89 | 2025-11 | Which Brazilian city hosted the COP30 climate summit in November 2025? | Belém / belem, belém | Belém | exact_alias_match | host_known_before_benchmark_date_range |
| 90 | 2025-11 | What was Anthropic&#x27;s flagship Claude model release of November 2025 called? | Claude Opus 4.5 / opus 4.5, 4.5 | Claude Opus 4.5 | exact_alias_match | none recorded |
| 91 | 2025-11-12 | The longest US federal government shutdown in history ended on November 12, 2025 when President Trump signed a funding bill. How many days did it last? | 43 / 43 | 43 days | manual_review | none recorded |
| 92 | 2025-11-18 | Which flagship AI model did Google release on November 18, 2025, topping the LMArena leaderboard and launching alongside the Antigravity IDE? | Gemini 3 / Gemini 3 | Gemini 3 | exact_alias_match | none recorded |
| 93 | 2025-12 | Who won Chile&#x27;s December 2025 presidential runoff? | José Antonio Kast / kast | UNKNOWN | explicit_unknown | none recorded |
| 94 | 2026-02 | Who won Super Bowl LX in February 2026? | Seattle Seahawks / seahawks, seattle | UNKNOWN | explicit_unknown | none recorded |
| 95 | 2026-02 | Which country topped the medal table at the 2026 Winter Olympics in Milan-Cortina? | Norway / norway | Norway | exact_alias_match | none recorded |
| 96 | 2026-02 | Which country won the gold medal in men&#x27;s ice hockey at the 2026 Winter Olympics? | United States / united states, usa, american | United States | exact_alias_match | none recorded |
| 97 | 2026-02 | Who won Album of the Year at the Grammy Awards held in February 2026? | Bad Bunny (Debí Tirar Más Fotos) / bad bunny, debi tirar, debí tirar | Bad Bunny | exact_alias_match | none recorded |
| 98 | 2026-02 | Who won the men&#x27;s singles title at the 2026 Australian Open? | Carlos Alcaraz / alcaraz | UNKNOWN | explicit_unknown | none recorded |
| 99 | 2026-02 | Who won Costa Rica&#x27;s presidential election held on February 1, 2026? | Laura Fernández / fernandez, fernández | UNKNOWN | explicit_unknown | none recorded |
| 100 | 2026-02 | Which country&#x27;s supreme leader was assassinated in airstrikes on his capital on February 28, 2026? | Iran (Ali Khamenei) / iran, khamenei | Iran | exact_alias_match | person_alias_for_country_question |
| 101 | 2026-02-15 | Which Oscar-winning actor, star of The Godfather and Apocalypse Now, died on February 15, 2026 at age 95? | Robert Duvall / Robert Duvall | Robert Duvall | exact_alias_match | publication_after_event_day |
| 102 | 2026-03 | Which film won Best Picture at the 98th Academy Awards in March 2026? | One Battle After Another / one battle after another | UNKNOWN | explicit_unknown | none recorded |
| 103 | 2026-03 | Which country won the 2026 Six Nations rugby championship? | France / france | UNKNOWN | explicit_unknown | none recorded |
| 104 | 2026-03 | Which version of its flagship GPT model series did OpenAI release on March 5, 2026? | GPT-5.4 / 5.4 | GPT-5.4 | exact_alias_match | none recorded |
| 105 | 2026-03 | Who was elected Iran&#x27;s third supreme leader in March 2026? | Mojtaba Khamenei / mojtaba | UNKNOWN | explicit_unknown | none recorded |
| 106 | 2026-03-15 | Who won Best Actor at the 98th Academy Awards on March 15, 2026? | Michael B. Jordan / Michael B. Jordan, Michael B. Jordan (for Sinners) | Michael B. Jordan | exact_alias_match | none recorded |
| 107 | 2026-04 | Who became Prime Minister of Hungary following the April 2026 parliamentary election? | Péter Magyar / magyar | UNKNOWN | explicit_unknown | election_month_is_not_taking_office_month |
| 108 | 2026-04 | Which company released the AI model GPT-5.5 in April 2026? | OpenAI / openai | UNKNOWN | explicit_unknown | brand_to_company_inference |
| 109 | 2026-04-06 | Which team won the 2026 NCAA men&#x27;s basketball national championship on April 6, 2026? | Michigan Wolverines / Michigan Wolverines, Michigan Wolverines (beat UConn 69-63) | UNKNOWN | explicit_unknown | none recorded |
| 110 | 2026-05 | Which horse won the 2026 Kentucky Derby? | Golden Tempo / golden tempo | UNKNOWN | explicit_unknown | none recorded |
| 111 | 2026-05 | Which country won the Eurovision Song Contest 2026? | Bulgaria / bulgaria | UNKNOWN | explicit_unknown | none recorded |
| 112 | 2026-05 | Which film won the Palme d&#x27;Or at the 2026 Cannes Film Festival? | Fjord / fjord, mungiu | UNKNOWN | explicit_unknown | director_alias_for_film_question |
| 113 | 2026-05 | Which club won the 2026 UEFA Champions League final in Budapest? | Paris Saint-Germain / paris, psg | UNKNOWN | explicit_unknown | city_only_club_alias |
| 114 | 2026-06 | Who won the men&#x27;s singles title at the 2026 French Open? | Alexander Zverev / zverev | UNKNOWN | explicit_unknown | none recorded |
| 115 | 2026-06 | Which team won the 2026 NBA Finals? | New York Knicks / knicks | UNKNOWN | explicit_unknown | none recorded |
| 116 | 2026-06 | Which team won the 2026 Stanley Cup Final? | Carolina Hurricanes / hurricanes, carolina | UNKNOWN | explicit_unknown | none recorded |
| 117 | 2026-06 | Who won Colombia&#x27;s presidential runoff election on June 21, 2026? | Abelardo de la Espriella / espriella, abelardo | UNKNOWN | explicit_unknown | original_source_is_preliminary_result |
| 118 | 2026-06-06 | Who won the 2026 French Open women&#x27;s singles title on June 6, 2026? | Mirra Andreeva / Mirra Andreeva, Mirra Andreeva (beat Maja Chwalinska) | UNKNOWN | explicit_unknown | none recorded |
| 119 | 2026-06-09 | Which flagship AI model did Anthropic release on June 9, 2026, alongside the restricted Mythos 5? | Claude Fable 5 / Claude Fable 5 | Claude Opus 5 | manual_review | none recorded |
| 120 | 2026-06-24 | Which country was struck by twin magnitude 7.2 and 7.5 earthquakes on June 24, 2026, its strongest since 1900, killing over 2,000 people? | Venezuela / Venezuela | UNKNOWN | explicit_unknown | time_varying_casualty_premise |
| 121 | 2026-07 | Which country won the 2026 FIFA World Cup final on July 19, 2026? | Spain / spain | UNKNOWN | explicit_unknown | none recorded |
| 122 | 2026-07 | Who won the men&#x27;s singles title at Wimbledon 2026? | Jannik Sinner / sinner | UNKNOWN | explicit_unknown | none recorded |
| 123 | 2026-07 | Who won the women&#x27;s singles title at Wimbledon 2026? | Linda Nosková / noskova, nosková | UNKNOWN | explicit_unknown | none recorded |
| 124 | 2026-07 | Who won the 2026 Tour de France general classification? | Tadej Pogačar / pogacar, pogačar | UNKNOWN | explicit_unknown | none recorded |
| 125 | 2026-07 | Who won the 2026 Open Championship (British Open) at Royal Birkdale? | Ryan Fox / ryan fox | UNKNOWN | explicit_unknown | none recorded |
| 126 | 2026-07 | Which company released the AI model Claude Opus 5 on July 24, 2026? | Anthropic / anthropic | Anthropic | exact_alias_match | brand_to_company_inference |
| 127 | 2026-07 | Which version of its flagship GPT model series did OpenAI release on July 9, 2026? | GPT-5.6 / 5.6 | UNKNOWN | explicit_unknown | limited_preview_precedes_public_release |
| 128 | 2026-08 | Which country topped the medal table at the 2026 Commonwealth Games in Glasgow? | Australia / australia | UNKNOWN | explicit_unknown | none recorded |
| 129 | 2026-08 | Which actress, star of Heroes and Nashville, died on August 16, 2026, at age 36? | Hayden Panettiere / panettiere | UNKNOWN | explicit_unknown | none recorded |

</details>

## Audit evidence

Catalog SHA-256: `926436aee906283794c0f97bfd8e8432f97e9fdcd9bb6f7d4ae46fdb3ef6c40b`.

Source audit SHA-256: `9d9b2700142adb8d5f609e333fcd7437881b7cdea0feb8d48bfb0431634c6ba9`.

The JSON contains full attempt histories, original-key grades, source evidence/flags, unapplied proposed corrections, completion/no-tools/model metadata, timestamps, usage counters and private receipt hashes. Session/request IDs, account/profile names, local paths, commands, process IDs and raw stdout/stderr are omitted. Private receipt/file hashes preserve audit bindings; named numeric usage evidence is retained. Unexpected reported model names are hashed instead of published.

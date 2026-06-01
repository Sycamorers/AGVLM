# SFT Format Audit

- Manifest: `/blue/hmedeiros/qinruoyao/agvlm/data/manifests/full/sft_train_phi4_max3_closed_label_classification_repair.jsonl`
- Prompt format: `instructional`
- Target format: `instructional`
- Unique examples rendered: `32`
- Validation failures: `0`

## Counts

| Task | Manifest rows | Rendered unique examples |
| --- | ---: | ---: |
| clarify_or_respond | 6482 | 8 |
| classification | 86016 | 8 |
| consultation | 25000 | 8 |
| vqa | 50000 | 8 |

## clarify_or_respond

### clarify_or_respond.1 `mirage-#828224`

- Source: `mirage`
- Verifier: `clarify`
- Images: `data/raw/mirage/full/images/MMMT_Decomp/train/MMMT_Decomp-train-003513-01.png`, `data/raw/mirage/full/images/MMMT_Decomp/train/MMMT_Decomp-train-003513-02.png`

**Prompt**

```text
user: What could have made this hole in the tree?
expert: Hi, Reese looks like it is from a pileated woodpecker, searching for insects in the wood. They are beautiful birds and if you listen for their typical 'rat-tat-tat' hunting on the tree, you may be able to catch a glimpse from your window, or slowly sneak up on it in the yard.  Unfortunately that usually means that the tree is nearing its end stage of life since there are insects inhabiting it. When the beetles and borers move into a tree it is typically a secondary response, either old trees, or another stressor causing decline and decaying wood. You can look up into the canopy and see if you notice any dead or dying sections as well. Is this a white pine? (it looks like the bark) if so, it would show a large sections of orange needles - beyond the typical spring needle drop.  If the canopy still looks to be in good shape, it could be that the wood pecker is just doing something territorial. You could try wrapping a little hardware cloth around the trunk to discourage it, although if there are insects in there and the woodpecker has found a viable food source, it will just find another area on the trunk to poke at.  Feel free to send us more photos of the canopy or if you notice other exit holes from wood boring insects. There isn't much you can do for the insects at this point, but if you have concern of the tree falling and damaging structures, you may want to consult with a certified arborist for removal. The link provided will help you find one for your area.  Emily
user: Thank you for the helpful information. It's not a pine tree.  Attached is a picture of the canopy.  Thanks again.Tori CrichlowOn Wed, May 3, 2023 at 12:32 PM Ask Extension <<personal data hidden>> wrote:

Respond using exactly one of these formats:
Decision: clarify
Clarifying question: <one question needed before diagnosis or management>

Decision: respond
Answer: <concise agricultural answer>
```

**Target**

```text
Decision: clarify
Clarifying question: Thank you for the additional information and the image of the canopy. Could you describe what you observe in the canopy? Are there any dead or dying sections, or any other signs of decay or insect infestation? Additionally, are you concerned about the tree's stability or potential damage to nearby structures?
```

### clarify_or_respond.2 `mirage-#879622`

- Source: `mirage`
- Verifier: `clarify`
- Images: `data/raw/mirage/full/images/MMMT_Decomp/train/MMMT_Decomp-train-003825-01.png`, `data/raw/mirage/full/images/MMMT_Decomp/train/MMMT_Decomp-train-003825-02.png`

**Prompt**

```text
user: Can you help me identify two caterpillar-like creatures I found on my purple coneflower today? One is pretty normal looking, the other, of which there are about a half dozen, is really strange looking.Thanks!

Respond using exactly one of these formats:
Decision: clarify
Clarifying question: <one question needed before diagnosis or management>

Decision: respond
Answer: <concise agricultural answer>
```

**Target**

```text
Decision: respond
Answer: Based on your description and the information you've provided, it seems likely that the caterpillars you found are indeed the larvae of the Wavy Lined Emerald Moth. These caterpillars are known for their unique camouflage behavior, where they attach plant material to themselves to blend in with their surroundings. This is a fascinating adaptation that helps them avoid predators. If you want to confirm this identification, you can compare the caterpillars in your images to those in the article you mentioned. If they match, you can be quite confident in your identification. If you have any further questions or need more assistance, feel free to ask!
```

### clarify_or_respond.3 `mirage-#883094`

- Source: `mirage`
- Verifier: `clarify`
- Images: `data/raw/mirage/full/images/MMMT_Decomp/train/MMMT_Decomp-train-002259-01.png`, `data/raw/mirage/full/images/MMMT_Decomp/train/MMMT_Decomp-train-002259-02.png`

**Prompt**

```text
user: I have spruce ( I cannot remember if it is the Norway or Blue, but it looks like Blue from the photo) that have some deformities and die-off at the end of some branches. Can you please help to identify and suggest possible treatments? Oceana county.

Respond using exactly one of these formats:
Decision: clarify
Clarifying question: <one question needed before diagnosis or management>

Decision: respond
Answer: <concise agricultural answer>
```

**Target**

```text
Decision: respond
Answer: Based on the information provided, it seems like you have a comprehensive understanding of the issues affecting your spruce trees. The Cooley Spruce Gall Adelgid is known to cause the symptoms you described, such as deformities and die-off at the ends of branches. Your plan to physically remove any visible signs of disease is a good start. Treating with a pyrethroid in the spring is also a recommended approach for managing adelgids. Additionally, maintaining the health of your trees through proper watering, mulching, and fertilization can help them withstand pest pressures. Since you have already seen success with Actinovate on your Douglas Fir, continuing that treatment seems wise. Keep monitoring your trees for any new signs of distress and consider consulting with a local arborist for ongoing management strategies. If you have any further questions or need more specific advice, feel free to reach out.
```

### clarify_or_respond.4 `mirage-#883161`

- Source: `mirage`
- Verifier: `clarify`
- Images: `data/raw/mirage/full/images/MMMT_Direct/train/MMMT_Direct-train-001177-01.png`, `data/raw/mirage/full/images/MMMT_Direct/train/MMMT_Direct-train-001177-02.png`

**Prompt**

```text
user: I would like to send pictures of Flys that are taking over my street

Respond using exactly one of these formats:
Decision: clarify
Clarifying question: <one question needed before diagnosis or management>

Decision: respond
Answer: <concise agricultural answer>
```

**Target**

```text
Decision: clarify
Clarifying question: Could you provide more details about the environment where these flies are prevalent, such as the climate and any specific areas they are concentrated in? Additionally, have you tried any control measures so far?
```

### clarify_or_respond.5 `mirage-#837713`

- Source: `mirage`
- Verifier: `clarify`
- Images: `data/raw/mirage/full/images/MMMT_Direct/train/MMMT_Direct-train-003512-01.png`, `data/raw/mirage/full/images/MMMT_Direct/train/MMMT_Direct-train-003512-02.png`, `data/raw/mirage/full/images/MMMT_Direct/train/MMMT_Direct-train-003512-03.png`

**Prompt**

```text
user: Backyard garden in Portland.  All of a sudden, leaves on our fava bean, pea, and cucumber plants all have white-yellow spots, lines, and/or bruises -- along with torn leaves and holes. On 6/18, we had an intense hail storm (which even left ice on the ground overnight in some spots!). I left town the next day and didn't look carefully at anything in the garden again until today (6/26). Surely some of the physical damage is the hail, and it's been pretty hot since the storm too, but does that explain all of it?  Looking at various pests/diseases, mosaic virus seemed like it could be an explanation. Related: our blueberry bushes are also suddenly poor as well -- which I'll submit as a separate question, but mention here fwiw. Thank you.

Respond using exactly one of these formats:
Decision: clarify
Clarifying question: <one question needed before diagnosis or management>

Decision: respond
Answer: <concise agricultural answer>
```

**Target**

```text
Decision: clarify
Clarifying question: Could you please provide the images of the affected plants and let me know if you've noticed any pests or applied any treatments or fertilizers recently?
```

### clarify_or_respond.6 `mirage-#874239`

- Source: `mirage`
- Verifier: `clarify`
- Images: `data/raw/mirage/full/images/MMMT_Direct/train/MMMT_Direct-train-002925-01.png`, `data/raw/mirage/full/images/MMMT_Direct/train/MMMT_Direct-train-002925-02.png`

**Prompt**

```text
user: We have a large cedar tree in our sidewalk planting strip that was likely planted when our home was built over 110 years ago. The top of the tree has been losing its foliage (needles) the past few years which coincides with our much hotter and drier summers. We hav been drip irrigating our large street trees for the past 3 years but didn’t start until August when it became obvious the summers were going to be real dry. This year in early June I provided drip irrigation for two days hoping to satiate the soil. I’ve noticed that even during rainy months the soil remains dry due to the foliage cover acting as an umbrella.
Is this dead area atop the tree a sign the tree is doomed or will supplementary watering help save the tree?

Respond using exactly one of these formats:
Decision: clarify
Clarifying question: <one question needed before diagnosis or management>

Decision: respond
Answer: <concise agricultural answer>
```

**Target**

```text
Decision: respond
Answer: The loss of foliage at the top of your cedar tree could be a sign of stress, often due to environmental factors such as drought. Your approach of providing deep watering is a good step, especially since the soil remains dry even during rainy months. It's important to ensure that the water reaches deep into the root zone, as surface watering may not be sufficient. Consider extending the duration and frequency of your watering schedule, particularly during the dry months, to help the tree recover. Additionally, mulching around the base of the tree can help retain soil moisture. While it's difficult to predict the tree's recovery, your proactive measures can certainly improve its chances of regaining vigor. Keep monitoring the tree for any changes and consult with a local arborist if the condition worsens.
```

### clarify_or_respond.7 `mirage-#832090`

- Source: `mirage`
- Verifier: `clarify`
- Images: `data/raw/mirage/full/images/MMMT_Decomp/train/MMMT_Decomp-train-001492-01.png`, `data/raw/mirage/full/images/MMMT_Decomp/train/MMMT_Decomp-train-001492-02.png`

**Prompt**

```text
user: We have a hydrangea tree that is about 5 years old. It has always done well but for whatever reason this year the leaves are not growing properly.

Respond using exactly one of these formats:
Decision: clarify
Clarifying question: <one question needed before diagnosis or management>

Decision: respond
Answer: <concise agricultural answer>
```

**Target**

```text
Decision: respond
Answer: It sounds like there are a few factors that could be affecting your hydrangea tree. The change in water distribution due to the gutter being moved could have altered the moisture levels the tree is accustomed to. Hydrangeas generally prefer consistent moisture, so ensuring the soil remains evenly moist but not waterlogged is important. Additionally, starting a regular fertilization schedule can help provide the nutrients the tree needs to thrive. Be cautious with the type of fertilizer; a balanced, slow-release fertilizer formulated for flowering shrubs would be ideal. Lastly, if the area is a favorite spot for your dog and other dogs, consider protecting the base of the tree to prevent soil compaction and potential damage to the roots. With these adjustments, your hydrangea should hopefully return to its healthy state.
```

### clarify_or_respond.8 `mirage-#837686`

- Source: `mirage`
- Verifier: `clarify`
- Images: `data/raw/mirage/full/images/MMMT_Direct/train/MMMT_Direct-train-003014-01.png`

**Prompt**

```text
user: I am growing tomatoes in 15 gallon nursery pots like you see in the picture and wonder if I need to protect the pots from sunlight? Mats

Respond using exactly one of these formats:
Decision: clarify
Clarifying question: <one question needed before diagnosis or management>

Decision: respond
Answer: <concise agricultural answer>
```

**Target**

```text
Decision: respond
Answer: It sounds like you've taken some great initial steps to protect your tomato plants from excessive heat. The screens and mulch are both effective methods to reduce soil temperature and retain moisture. As temperatures rise to 100 degrees, continue to monitor the soil temperature and moisture levels. You might also consider watering early in the morning or late in the evening to minimize evaporation. If you notice any signs of stress on the plants, such as wilting or leaf scorch, you may need to adjust your protection methods or increase watering frequency. Keep an eye on the plants and adjust as necessary.
```


## classification

### classification.1 `plantvillage-train-042224`

- Source: `plantvillage`
- Verifier: `label`
- Images: `data/raw/plantvillage/full/images/train/042224.png`

**Prompt**

```text
Identify the crop issue or pest in this agricultural image.

Choose exactly one label from this allowed label set:
Allowed labels: apple apple scab; apple black rot; apple cedar apple rust; apple healthy; blueberry healthy; cherry including sour healthy; cherry including sour powdery mildew; corn maize cercospora leaf spot gray leaf spot; corn maize common rust; corn maize healthy; corn maize northern leaf blight; grape black rot; grape esca black measles; grape healthy; grape leaf blight isariopsis leaf spot; orange haunglongbing citrus greening; peach bacterial spot; peach healthy; pepper bell bacterial spot; pepper bell healthy; potato early blight; potato healthy; potato late blight; raspberry healthy; soybean healthy; squash powdery mildew; strawberry healthy; strawberry leaf scorch; tomato bacterial spot; tomato early blight; tomato healthy; tomato late blight; tomato leaf mold; tomato septoria leaf spot; tomato spider mites two spotted spider mite; tomato target spot; tomato tomato mosaic virus; tomato tomato yellow leaf curl virus
Respond in this format:
Answer: <one allowed label>
Evidence: <brief visible symptom evidence>
Do not leave Answer blank, invent labels, or copy the placeholder text.
```

**Target**

```text
Answer: tomato healthy
Evidence: Visible Tomato symptoms support the healthy label.
```

### classification.2 `ip102-ip102-v1-1-images-60933-jpg`

- Source: `ip102`
- Verifier: `label`
- Images: `data/raw/ip102/full/ip102_v1.1/images/60933.jpg`

**Prompt**

```text
Identify the insect or pest shown in this agricultural image.

Choose exactly one label from this allowed label set:
Allowed labels: Adristyrannus; Aleurocanthus spiniferus; alfalfa plant bug; alfalfa seed chalcid; alfalfa weevil; Ampelophaga; aphids; Aphis citricola Vander Goot; Apolygus lucorum; army worm; asiatic rice borer; Bactrocera tsuneonis; beet army worm; beet fly; Beet spot flies; beet weevil; bird cherry oataphid; black cutworm; blister beetle; Brevipoalpus lewisi McGregor; brown plant hopper; cabbage army worm; cerodonta denticornis; Ceroplastes rubens; Chlumetia transversa; Chrysomphalus aonidum; Cicadella viridis; Cicadellidae; Colomerus vitis; corn borer; Dacus dorsalis(Hendel); Dasineura sp; Deporaus marginatus Pascoe; english grain aphid; Erythroneura apicalis; flax budworm; flea beetle; grain spreader thrips; green bug; grub; Icerya purchasi Maskell; large cutworm; Lawana imitata Melichar; legume blister beetle; Limacodidae; Locustoidea; longlegged spider mite; Lycorma delicatula; lytta polita; Mango flat beak leafhopper; meadow moth; Miridae; mole cricket; Nipaecoccus vastalor; odontothrips loti; oides decempunctata; paddy stem maggot; Panonchus citri McGregor; Papilio xuthus; parathrene regalis; Parlatoria zizyphus Lucus; peach borer; penthaleus major; Phyllocnistis citrella Stainton; Phyllocoptes oleiverus ashmead; Pieris canidia; Polyphagotars onemus latus; Potosiabre vitarsis; Prodenia litura; Pseudococcus comstocki Kuwana; red spider; Rhytidodera bowrinii white; rice gall midge; rice leaf caterpillar; rice leaf roller; rice leafhopper; rice shell pest; Rice Stemfly; rice water weevil; Salurnis marginella Guerr; Scirtothrips dorsalis Hood; sericaorient alismots chulsky; small brown plant hopper; Sternochetus frigidus; tarnished plant bug; Tetradacus c Bactrocera minax; therioaphis maculata Buckton; Thrips; Toxoptera aurantii; Toxoptera citricidus; Trialeurodes vaporariorum; Unaspis yanonensis; Viteus vitifoliae; wheat blossom midge; wheat phloeothrips; wheat sawfly; white backed plant hopper; white margined moth; wireworm; Xylotrechus; yellow cutworm; yellow rice borer
Respond in this format:
Answer: <one allowed label>
Evidence: <brief visible symptom evidence>
Do not leave Answer blank, invent labels, or copy the placeholder text.
```

**Target**

```text
Answer: Nipaecoccus vastalor
Evidence: Visible agricultural symptoms or pest features support this label.
```

### classification.3 `ip102-ip102-v1-1-images-16091-jpg`

- Source: `ip102`
- Verifier: `label`
- Images: `data/raw/ip102/full/ip102_v1.1/images/16091.jpg`

**Prompt**

```text
Identify the insect or pest shown in this agricultural image.

Choose exactly one label from this allowed label set:
Allowed labels: Adristyrannus; Aleurocanthus spiniferus; alfalfa plant bug; alfalfa seed chalcid; alfalfa weevil; Ampelophaga; aphids; Aphis citricola Vander Goot; Apolygus lucorum; army worm; asiatic rice borer; Bactrocera tsuneonis; beet army worm; beet fly; Beet spot flies; beet weevil; bird cherry oataphid; black cutworm; blister beetle; Brevipoalpus lewisi McGregor; brown plant hopper; cabbage army worm; cerodonta denticornis; Ceroplastes rubens; Chlumetia transversa; Chrysomphalus aonidum; Cicadella viridis; Cicadellidae; Colomerus vitis; corn borer; Dacus dorsalis(Hendel); Dasineura sp; Deporaus marginatus Pascoe; english grain aphid; Erythroneura apicalis; flax budworm; flea beetle; grain spreader thrips; green bug; grub; Icerya purchasi Maskell; large cutworm; Lawana imitata Melichar; legume blister beetle; Limacodidae; Locustoidea; longlegged spider mite; Lycorma delicatula; lytta polita; Mango flat beak leafhopper; meadow moth; Miridae; mole cricket; Nipaecoccus vastalor; odontothrips loti; oides decempunctata; paddy stem maggot; Panonchus citri McGregor; Papilio xuthus; parathrene regalis; Parlatoria zizyphus Lucus; peach borer; penthaleus major; Phyllocnistis citrella Stainton; Phyllocoptes oleiverus ashmead; Pieris canidia; Polyphagotars onemus latus; Potosiabre vitarsis; Prodenia litura; Pseudococcus comstocki Kuwana; red spider; Rhytidodera bowrinii white; rice gall midge; rice leaf caterpillar; rice leaf roller; rice leafhopper; rice shell pest; Rice Stemfly; rice water weevil; Salurnis marginella Guerr; Scirtothrips dorsalis Hood; sericaorient alismots chulsky; small brown plant hopper; Sternochetus frigidus; tarnished plant bug; Tetradacus c Bactrocera minax; therioaphis maculata Buckton; Thrips; Toxoptera aurantii; Toxoptera citricidus; Trialeurodes vaporariorum; Unaspis yanonensis; Viteus vitifoliae; wheat blossom midge; wheat phloeothrips; wheat sawfly; white backed plant hopper; white margined moth; wireworm; Xylotrechus; yellow cutworm; yellow rice borer
Respond in this format:
Answer: <one allowed label>
Evidence: <brief visible symptom evidence>
Do not leave Answer blank, invent labels, or copy the placeholder text.
```

**Target**

```text
Answer: army worm
Evidence: Visible agricultural symptoms or pest features support this label.
```

### classification.4 `plantvillage-train-010403`

- Source: `plantvillage`
- Verifier: `label`
- Images: `data/raw/plantvillage/full/images/train/010403.png`

**Prompt**

```text
Identify the crop issue or pest in this agricultural image.

Choose exactly one label from this allowed label set:
Allowed labels: apple apple scab; apple black rot; apple cedar apple rust; apple healthy; blueberry healthy; cherry including sour healthy; cherry including sour powdery mildew; corn maize cercospora leaf spot gray leaf spot; corn maize common rust; corn maize healthy; corn maize northern leaf blight; grape black rot; grape esca black measles; grape healthy; grape leaf blight isariopsis leaf spot; orange haunglongbing citrus greening; peach bacterial spot; peach healthy; pepper bell bacterial spot; pepper bell healthy; potato early blight; potato healthy; potato late blight; raspberry healthy; soybean healthy; squash powdery mildew; strawberry healthy; strawberry leaf scorch; tomato bacterial spot; tomato early blight; tomato healthy; tomato late blight; tomato leaf mold; tomato septoria leaf spot; tomato spider mites two spotted spider mite; tomato target spot; tomato tomato mosaic virus; tomato tomato yellow leaf curl virus
Respond in this format:
Answer: <one allowed label>
Evidence: <brief visible symptom evidence>
Do not leave Answer blank, invent labels, or copy the placeholder text.
```

**Target**

```text
Answer: grape esca black measles
Evidence: Visible Grape symptoms support the Esca (Black Measles) label.
```

### classification.5 `plantvillage-train-002585`

- Source: `plantvillage`
- Verifier: `label`
- Images: `data/raw/plantvillage/full/images/train/002585.png`

**Prompt**

```text
Identify the crop issue or pest in this agricultural image.

Choose exactly one label from this allowed label set:
Allowed labels: apple apple scab; apple black rot; apple cedar apple rust; apple healthy; blueberry healthy; cherry including sour healthy; cherry including sour powdery mildew; corn maize cercospora leaf spot gray leaf spot; corn maize common rust; corn maize healthy; corn maize northern leaf blight; grape black rot; grape esca black measles; grape healthy; grape leaf blight isariopsis leaf spot; orange haunglongbing citrus greening; peach bacterial spot; peach healthy; pepper bell bacterial spot; pepper bell healthy; potato early blight; potato healthy; potato late blight; raspberry healthy; soybean healthy; squash powdery mildew; strawberry healthy; strawberry leaf scorch; tomato bacterial spot; tomato early blight; tomato healthy; tomato late blight; tomato leaf mold; tomato septoria leaf spot; tomato spider mites two spotted spider mite; tomato target spot; tomato tomato mosaic virus; tomato tomato yellow leaf curl virus
Respond in this format:
Answer: <one allowed label>
Evidence: <brief visible symptom evidence>
Do not leave Answer blank, invent labels, or copy the placeholder text.
```

**Target**

```text
Answer: blueberry healthy
Evidence: Visible Blueberry symptoms support the healthy label.
```

### classification.6 `plantvillage-train-043444`

- Source: `plantvillage`
- Verifier: `label`
- Images: `data/raw/plantvillage/full/images/train/043444.png`

**Prompt**

```text
Identify the crop issue or pest in this agricultural image.

Choose exactly one label from this allowed label set:
Allowed labels: apple apple scab; apple black rot; apple cedar apple rust; apple healthy; blueberry healthy; cherry including sour healthy; cherry including sour powdery mildew; corn maize cercospora leaf spot gray leaf spot; corn maize common rust; corn maize healthy; corn maize northern leaf blight; grape black rot; grape esca black measles; grape healthy; grape leaf blight isariopsis leaf spot; orange haunglongbing citrus greening; peach bacterial spot; peach healthy; pepper bell bacterial spot; pepper bell healthy; potato early blight; potato healthy; potato late blight; raspberry healthy; soybean healthy; squash powdery mildew; strawberry healthy; strawberry leaf scorch; tomato bacterial spot; tomato early blight; tomato healthy; tomato late blight; tomato leaf mold; tomato septoria leaf spot; tomato spider mites two spotted spider mite; tomato target spot; tomato tomato mosaic virus; tomato tomato yellow leaf curl virus
Respond in this format:
Answer: <one allowed label>
Evidence: <brief visible symptom evidence>
Do not leave Answer blank, invent labels, or copy the placeholder text.
```

**Target**

```text
Answer: tomato healthy
Evidence: Visible Tomato symptoms support the healthy label.
```

### classification.7 `ip102-ip102-v1-1-images-43817-jpg`

- Source: `ip102`
- Verifier: `label`
- Images: `data/raw/ip102/full/ip102_v1.1/images/43817.jpg`

**Prompt**

```text
Identify the insect or pest shown in this agricultural image.

Choose exactly one label from this allowed label set:
Allowed labels: Adristyrannus; Aleurocanthus spiniferus; alfalfa plant bug; alfalfa seed chalcid; alfalfa weevil; Ampelophaga; aphids; Aphis citricola Vander Goot; Apolygus lucorum; army worm; asiatic rice borer; Bactrocera tsuneonis; beet army worm; beet fly; Beet spot flies; beet weevil; bird cherry oataphid; black cutworm; blister beetle; Brevipoalpus lewisi McGregor; brown plant hopper; cabbage army worm; cerodonta denticornis; Ceroplastes rubens; Chlumetia transversa; Chrysomphalus aonidum; Cicadella viridis; Cicadellidae; Colomerus vitis; corn borer; Dacus dorsalis(Hendel); Dasineura sp; Deporaus marginatus Pascoe; english grain aphid; Erythroneura apicalis; flax budworm; flea beetle; grain spreader thrips; green bug; grub; Icerya purchasi Maskell; large cutworm; Lawana imitata Melichar; legume blister beetle; Limacodidae; Locustoidea; longlegged spider mite; Lycorma delicatula; lytta polita; Mango flat beak leafhopper; meadow moth; Miridae; mole cricket; Nipaecoccus vastalor; odontothrips loti; oides decempunctata; paddy stem maggot; Panonchus citri McGregor; Papilio xuthus; parathrene regalis; Parlatoria zizyphus Lucus; peach borer; penthaleus major; Phyllocnistis citrella Stainton; Phyllocoptes oleiverus ashmead; Pieris canidia; Polyphagotars onemus latus; Potosiabre vitarsis; Prodenia litura; Pseudococcus comstocki Kuwana; red spider; Rhytidodera bowrinii white; rice gall midge; rice leaf caterpillar; rice leaf roller; rice leafhopper; rice shell pest; Rice Stemfly; rice water weevil; Salurnis marginella Guerr; Scirtothrips dorsalis Hood; sericaorient alismots chulsky; small brown plant hopper; Sternochetus frigidus; tarnished plant bug; Tetradacus c Bactrocera minax; therioaphis maculata Buckton; Thrips; Toxoptera aurantii; Toxoptera citricidus; Trialeurodes vaporariorum; Unaspis yanonensis; Viteus vitifoliae; wheat blossom midge; wheat phloeothrips; wheat sawfly; white backed plant hopper; white margined moth; wireworm; Xylotrechus; yellow cutworm; yellow rice borer
Respond in this format:
Answer: <one allowed label>
Evidence: <brief visible symptom evidence>
Do not leave Answer blank, invent labels, or copy the placeholder text.
```

**Target**

```text
Answer: parathrene regalis
Evidence: Visible agricultural symptoms or pest features support this label.
```

### classification.8 `plantvillage-train-006228`

- Source: `plantvillage`
- Verifier: `label`
- Images: `data/raw/plantvillage/full/images/train/006228.png`

**Prompt**

```text
Identify the crop issue or pest in this agricultural image.

Choose exactly one label from this allowed label set:
Allowed labels: apple apple scab; apple black rot; apple cedar apple rust; apple healthy; blueberry healthy; cherry including sour healthy; cherry including sour powdery mildew; corn maize cercospora leaf spot gray leaf spot; corn maize common rust; corn maize healthy; corn maize northern leaf blight; grape black rot; grape esca black measles; grape healthy; grape leaf blight isariopsis leaf spot; orange haunglongbing citrus greening; peach bacterial spot; peach healthy; pepper bell bacterial spot; pepper bell healthy; potato early blight; potato healthy; potato late blight; raspberry healthy; soybean healthy; squash powdery mildew; strawberry healthy; strawberry leaf scorch; tomato bacterial spot; tomato early blight; tomato healthy; tomato late blight; tomato leaf mold; tomato septoria leaf spot; tomato spider mites two spotted spider mite; tomato target spot; tomato tomato mosaic virus; tomato tomato yellow leaf curl virus
Respond in this format:
Answer: <one allowed label>
Evidence: <brief visible symptom evidence>
Do not leave Answer blank, invent labels, or copy the placeholder text.
```

**Target**

```text
Answer: corn maize common rust
Evidence: Visible Corn (maize) symptoms support the Common rust label.
```


## consultation

### consultation.1 `agbase-agbase-846818`

- Source: `agbase`
- Verifier: `structured`
- Images: `data/raw/agbase/full/846818/846818_2.jpg`, `data/raw/agbase/full/846818/846818_3.jpg`, `data/raw/agbase/full/846818/846818_1.jpg`

**Prompt**

```text
Provide an expert agricultural diagnosis and management plan for the issue shown.

Respond using these line-start section headers exactly once:
Diagnosis:
Evidence:
Uncertainty:
Management:
Follow-up:
```

**Target**

```text
Diagnosis: scorch or sunburn
Evidence: brown spots on mostly new growth
Uncertainty: Image-only assessment; confirm with field context before treatment.
Management: No source-specific management step was provided; recommend local extension follow-up.
Follow-up: Share close-up images and crop history if symptoms progress or the diagnosis is uncertain.
```

### consultation.2 `agbase-agbase-649758`

- Source: `agbase`
- Verifier: `structured`
- Images: `data/raw/agbase/full/649758/649758_2.jpg`, `data/raw/agbase/full/649758/649758_1.jpg`

**Prompt**

```text
Provide an expert agricultural diagnosis and management plan for the issue shown.

Respond using these line-start section headers exactly once:
Diagnosis:
Evidence:
Uncertainty:
Management:
Follow-up:
```

**Target**

```text
Diagnosis: spiderwort
Evidence: Visible symptoms should be checked against the image and crop context.
Uncertainty: Image-only assessment; confirm with field context before treatment.
Management: using herbicides like crossbow can help get rid of aggressive weeds. spray in the morning or evening when insects are less active. use a piece of plastic or cardboard as a shield to protect the plants you want to keep as you are spraying.
Follow-up: Share close-up images and crop history if symptoms progress or the diagnosis is uncertain.
```

### consultation.3 `agbase-agbase-647781`

- Source: `agbase`
- Verifier: `structured`
- Images: `data/raw/agbase/full/647781/647781_2.jpg`, `data/raw/agbase/full/647781/647781_3.jpg`, `data/raw/agbase/full/647781/647781_1.jpg`

**Prompt**

```text
Provide an expert agricultural diagnosis and management plan for the issue shown.

Respond using these line-start section headers exactly once:
Diagnosis:
Evidence:
Uncertainty:
Management:
Follow-up:
```

**Target**

```text
Diagnosis: double freeze events causing branch dieback, leaf distortion, leaf and needle scorch
Evidence: needles on the branches are turning brown/rust color
Uncertainty: Image-only assessment; confirm with field context before treatment.
Management: water the root zone to a depth of 12-18 inches every couple of weeks when there hasn	been rain or snow, hold off on fertilizing until the tree is fully leafed out again, prune out dead branches after new growth has emerged, add 3-4 inches of mulch under the tree out to the drip line to moderate moisture and temperature changes around the roots
Follow-up: Share close-up images and crop history if symptoms progress or the diagnosis is uncertain.
```

### consultation.4 `agbase-agbase-873996`

- Source: `agbase`
- Verifier: `structured`
- Images: `data/raw/agbase/full/873996/873996_3.jpg`, `data/raw/agbase/full/873996/873996_2.jpg`, `data/raw/agbase/full/873996/873996_1.jpg`

**Prompt**

```text
Provide an expert agricultural diagnosis and management plan for the issue shown.

Respond using these line-start section headers exactly once:
Diagnosis:
Evidence:
Uncertainty:
Management:
Follow-up:
```

**Target**

```text
Diagnosis: ash tree
Evidence: two dead branches on the lower right side
Uncertainty: Image-only assessment; confirm with field context before treatment.
Management: No source-specific management step was provided; recommend local extension follow-up.
Follow-up: Share close-up images and crop history if symptoms progress or the diagnosis is uncertain.
```

### consultation.5 `agbase-agbase-822363`

- Source: `agbase`
- Verifier: `structured`
- Images: `data/raw/agbase/full/822363/822363_1.jpg`

**Prompt**

```text
Provide an expert agricultural diagnosis and management plan for the issue shown.

Respond using these line-start section headers exactly once:
Diagnosis:
Evidence:
Uncertainty:
Management:
Follow-up:
```

**Target**

```text
Diagnosis: rabbit damage
Evidence: bark devoured by rabbits
Uncertainty: Image-only assessment; confirm with field context before treatment.
Management: remove any loose bark or wood from around the wound. leave the wound exposed to the air and let nature take its course. if the damage is extensive, corrective pruning, regular watering and fertilization can help keep the tree healthy until it recovers.
Follow-up: Share close-up images and crop history if symptoms progress or the diagnosis is uncertain.
```

### consultation.6 `mirage-#831111`

- Source: `mirage`
- Verifier: `exact_match`
- Images: `data/raw/mirage/full/images/MMST_Standard/train/MMST_Standard-train-009767-01.png`, `data/raw/mirage/full/images/MMST_Standard/train/MMST_Standard-train-009767-02.png`

**Prompt**

```text
Please answer the following user question. You should first analyze the provided image, mentioning any visible clues or observations. Then, present the identification result. Write the entire content as one coherent paragraph (analysis + results).
User: What is invading my raised beds? I have a vine like plant that is invading raised beds and onion/garlic beds. It sends out rhizomes and even survives under black plastic inoculation?

Respond using these line-start section headers exactly once:
Diagnosis:
Evidence:
Uncertainty:
Management:
Follow-up:
```

**Target**

```text
Diagnosis: The images show a vigorous, sprawling vine with sprawling growth spreading over soil covered by black plastic mulch and weaving through onion or garlic plants in raised beds. The vine has green, somewhat oval to slightly arrowhead-shaped leaves and trailing stems that appear to root and spread via underground structures, consistent with rhizomes. Its ability to penetrate and survive beneath black plastic mulch highlights a robust underground network typical of invasive perennial vines. Based on these visible characteristics and the description of its vine-like growth, rhizomatous spread, and persistence in cultivated beds, the invasive plant is indeed bindweed, most likely field bindweed (Convolvulus arvensis) or hedge bindweed (Calystegia sepium). Bindweed is notorious for its tough rhizomes and deep roots that evade many control methods, making it a difficult-to-eradicate weed in garden beds and around onions and garlic. Managing this persistent invader requires thorough removal of all root fragments, vigilant monitoring to prevent flowering and seed set, cultural control practices, and, where safe and appropriate, cautious use of systemic herbicides.
Evidence: Visible symptoms should be checked against the image and crop context.
Uncertainty: Image-only assessment; confirm with field context before treatment.
Management: No source-specific management step was provided; recommend local extension follow-up.
Follow-up: Share close-up images and crop history if symptoms progress or the diagnosis is uncertain.
```

### consultation.7 `mirage-#869799`

- Source: `mirage`
- Verifier: `exact_match`
- Images: `data/raw/mirage/full/images/MMST_Standard/train/MMST_Standard-train-000287-01.png`, `data/raw/mirage/full/images/MMST_Standard/train/MMST_Standard-train-000287-02.png`, `data/raw/mirage/full/images/MMST_Standard/train/MMST_Standard-train-000287-03.png`

**Prompt**

```text
Please answer the following user question. You should first analyze the provided image, mentioning any visible clues or observations. Then, present the identification result. Write the entire content as one coherent paragraph (analysis + results).
User: What type of grass is this in my lawn? I have my front and back lawn seeded with 100% rye. I am afraid there is another grass overtaking the rye which is giving my lawn a two tone color. My hunch is that it is Poa Annua but I find it very difficult to identify. Perhaps it is just mature perennial rye?

Respond using these line-start section headers exactly once:
Diagnosis:
Evidence:
Uncertainty:
Management:
Follow-up:
```

**Target**

```text
Diagnosis: The images display a lawn with two distinct grass textures and colors. The broader, coarser grass blades exhibiting a darker, rougher green are indicative of tall fescue (Festuca arundinacea Schreb.), which is known for its coarse texture and darker hue compared to ryegrass. The finer, softer, and lighter green blades in the close-up — which are shown both in the lawn and being held by hand — correspond to perennial ryegrass, distinguished by its narrow leaves with a soft, smooth texture. The tall fescue’s coarser leaf blades may cause the white strands seen when mowing with a dull blade, as described in the conversation. This explains the two-tone coloration in your lawn since tall fescue is more robust and persistent compared to ryegrass. It is quite challenging to selectively remove tall fescue from ryegrass once established, and typically, the best approach to restoring a uniform ryegrass lawn is to remove the existing turf mechanically (such as with a sod cutter) and then reseed or resod with your preferred grass type.
Evidence: Visible symptoms should be checked against the image and crop context.
Uncertainty: Image-only assessment; confirm with field context before treatment.
Management: No source-specific management step was provided; recommend local extension follow-up.
Follow-up: Share close-up images and crop history if symptoms progress or the diagnosis is uncertain.
```

### consultation.8 `agbase-agbase-579836`

- Source: `agbase`
- Verifier: `structured`
- Images: `data/raw/agbase/full/579836/579836_1.jpg`, `data/raw/agbase/full/579836/579836_2.jpg`, `data/raw/agbase/full/579836/579836_3.jpg`

**Prompt**

```text
Provide an expert agricultural diagnosis and management plan for the issue shown.

Respond using these line-start section headers exactly once:
Diagnosis:
Evidence:
Uncertainty:
Management:
Follow-up:
```

**Target**

```text
Diagnosis: powdery mildew
Evidence: Visible symptoms should be checked against the image and crop context.
Uncertainty: Image-only assessment; confirm with field context before treatment.
Management: control with a fungicide from a local garden store, or reduce sprinkler irrigation to reduce humidity
Follow-up: Share close-up images and crop history if symptoms progress or the diagnosis is uncertain.
```


## vqa

### vqa.1 `plantvillage_vqa-image_008184.JPG-183128`

- Source: `plantvillage_vqa`
- Verifier: `exact_match`
- Images: `data/raw/plantvillage_vqa/full/images/train/image_008184.JPG`

**Prompt**

```text
Can you identify Tomato Yellow Leaf Curl Virus on this specimen?

Respond in this format:
Answer: <Yes or No>
```

**Target**

```text
Answer: Yes
```

### vqa.2 `plantvillage_vqa-image_020327.JPG-185752`

- Source: `plantvillage_vqa`
- Verifier: `exact_match`
- Images: `data/raw/plantvillage_vqa/full/images/train/image_020327.JPG`

**Prompt**

```text
Can you identify Huanglongbing infection indicators on this orange foliage?

Respond in this format:
Answer: <Yes or No>
```

**Target**

```text
Answer: Yes
```

### vqa.3 `plantvillage_vqa-image_035630.JPG-104782`

- Source: `plantvillage_vqa`
- Verifier: `exact_match`
- Images: `data/raw/plantvillage_vqa/full/images/train/image_035630.JPG`

**Prompt**

```text
Does this foliage match the characteristics of tomato plants?

Respond in this format:
Answer: <Yes or No>
```

**Target**

```text
Answer: Yes
```

### vqa.4 `plantvillage_vqa-image_041872.JPG-123119`

- Source: `plantvillage_vqa`
- Verifier: `exact_match`
- Images: `data/raw/plantvillage_vqa/full/images/train/image_041872.JPG`

**Prompt**

```text
Is this a Soybean leaf?

Respond in this format:
Answer: <Yes or No>
```

**Target**

```text
Answer: Yes
```

### vqa.5 `plantvillage_vqa-image_031600.JPG-092984`

- Source: `plantvillage_vqa`
- Verifier: `exact_match`
- Images: `data/raw/plantvillage_vqa/full/images/train/image_031600.JPG`

**Prompt**

```text
Are characteristics present that indicate pathogen absence?

Respond in this format:
Answer: <Yes or No>
```

**Target**

```text
Answer: No
```

### vqa.6 `plantvillage_vqa-image_027155.JPG-079790`

- Source: `plantvillage_vqa`
- Verifier: `exact_match`
- Images: `data/raw/plantvillage_vqa/full/images/train/image_027155.JPG`

**Prompt**

```text
Are characteristics present that indicate pathogen absence?

Respond in this format:
Answer: <Yes or No>
```

**Target**

```text
Answer: Yes
```

### vqa.7 `plantvillage_vqa-image_015346.JPG-044900`

- Source: `plantvillage_vqa`
- Verifier: `exact_match`
- Images: `data/raw/plantvillage_vqa/full/images/train/image_015346.JPG`

**Prompt**

```text
Picture this plant in optimal condition - how would its appearance alter?

Respond in this format:
Answer: <short answer>
```

**Target**

```text
Answer: A healthy version of this plant would not exhibit the evidence of orange or brown pustular structures.
```

### vqa.8 `plantvillage_vqa-image_017748.JPG-051920`

- Source: `plantvillage_vqa`
- Verifier: `exact_match`
- Images: `data/raw/plantvillage_vqa/full/images/train/image_017748.JPG`

**Prompt**

```text
Provide the taxonomic classification of this plant specimen.

Respond in this format:
Answer: <short answer>
```

**Target**

```text
Answer: Tomato
```

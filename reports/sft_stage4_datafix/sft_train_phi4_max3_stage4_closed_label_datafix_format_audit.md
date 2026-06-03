# SFT Format Audit

- Manifest: `/blue/hmedeiros/qinruoyao/agvlm/data/manifests/full/sft_train_phi4_max3_stage4_closed_label_datafix.jsonl`
- Prompt format: `instructional`
- Target format: `instructional`
- Unique examples rendered: `32`
- Validation failures: `0`

## Counts

| Task | Manifest rows | Rendered unique examples |
| --- | ---: | ---: |
| clarify_or_respond | 6482 | 8 |
| classification | 46848 | 8 |
| consultation | 25000 | 8 |
| vqa | 50000 | 8 |

## clarify_or_respond

### clarify_or_respond.1 `mirage-#838611`

- Source: `mirage`
- Verifier: `clarify`
- Images: `data/raw/mirage/full/images/MMMT_Decomp/train/MMMT_Decomp-train-000334-01.png`

**Prompt**

```text
user: Can you tell me what this insect is? I sprayed my tree just before budding and a few weeks later, which delayed the arrival of them. I've had them for several years now.

Respond using exactly one of these formats:
Decision: clarify
Clarifying question: <one question needed before diagnosis or management>

Decision: respond
Answer: <concise agricultural answer>
```

**Target**

```text
Decision: clarify
Clarifying question: Could you please provide more details about the type of tree affected and any other symptoms you might have observed? Additionally, if you could describe or confirm the contents of the attached image, that would be helpful.
```

### clarify_or_respond.2 `mirage-#830124`

- Source: `mirage`
- Verifier: `clarify`
- Images: `data/raw/mirage/full/images/MMMT_Decomp/train/MMMT_Decomp-train-000968-01.png`, `data/raw/mirage/full/images/MMMT_Decomp/train/MMMT_Decomp-train-000968-02.png`, `data/raw/mirage/full/images/MMMT_Decomp/train/MMMT_Decomp-train-000968-03.png`

**Prompt**

```text
user: Almost all my indoor and outdoor plants are infected with some kind of fungal disease. I’m sending photos of a miniature garden in a whiskey barrel that are most severely infected. The plants are( in the order of pics): thymus ‘Elfin’, dianthus gratianopolitanus ‘Tiny Rubies’, miniature daisies (Bellium Minutum). At the end of last September, I noticed that underneath the thyme it was severely black but the top layer of thyme was green, flowery and healthy looking. I also noticed a sludge looking layer underneath the Tiny Rubies but again the top was flowering profusely and otherwise healthy looking. Now this fall it’s the same with otherwise healthy flowers on top but the layer of the plant above the dirt is black or sludge-like. I sprinkled sulfur powder on this layer hoping to kill the fungi Al disease but now I see orange, yellow and black spots on all my other plants. I’ve been praying all my plants with fungicide to stop the spread but the chemical burnt many of my succulents that were healthy and all the rest are infected. I cannot cut the top layer of all my plants off or it will kill them all and I’m talking about thousands of dollars of plants. Even my grass has brown circles and vines growing on my trees are severely black and orange and yellow. I feel like it’s everywhere. Please help! I can’t tell what disease it is as it appears different on every different plant.
expert: It's unlikely a single disease is affecting such a wide variety of plants (a few pathogens have a wide host plant range, but many don't), but we'd need to see the symptoms on the other plants in order to try to diagnose the issue. You can send additional photos in your replies, and if the system limits you to three image files per reply, just reply more than once in order to share additional photos.  For the plants pictured, this type of mushy, dark foliage layer underneath the uppermost stems is fairly typical. (Not ideal, granted, but common enough it's not an unexpected issue.) Low-growing plants that require excellent drainage (as these three species do) and air circulation, both of which might be affected by the barrel planting where its rim reduces air circulation and the large soil volume is slower to drain excess moisture. Depending on their ingredients, some potting mixes drain better than others, though adjustments can be made in some cases by adding drainage-boosting ingredients like perlite, horticultural charcoal, and small expanded slate stones (sold as vole-discouraging amendments, often). Watering should only be done when the soil around several inches down is becoming dry to the touch, as over-watering is a common cause of decline that looks like this. As these lower leaves become covered-up by younger growth it's normal for them (especially on flattened creeping/sprawling plants) to shed by dying and turning brown, and their degradation by soil microbes and such occurs more rapidly when they stay damp, as appears to be the case here. Root rot, which happens when roots don't have enough access to oxygen in the soil due to wetness and succumb to opportunistic infection, and crown rot, when similar decay occurs on stems at the soil surface, is not treatable.  The only preventative measures that work well are using very well-drained soil or potting mix, giving plants good air circulation by avoiding crowding, and (in some cases) trying a grit-like stone mulch around the plant's root zone so the branches lying on the soil surface aren't in direct contact with the ground. (This kind of setup is used in rock gardens, for instance, which is what many of these dainty mini-groundcover-type species are used for. They also are often best grown in climates cooler and drier than ours, thought not exclusively.)  Sulfur is sometimes an effective fungicide but in this case it's not a useful treatment approach and might (over time) also alter the soil acidity to a level too low for these plants to thrive. Very few chemical fungicides are labeled for use in suppressing root and crown rot, but they are more expensive than most other fungicides, require reapplication to continually suppress fungal growth, and cannot cure existing infection (which may have already taken hold before symptoms manifested). As such, and because fungicides sometimes risk harming other organisms, we generally do not recommend their use. If certain plant species tend to contract recurring problems like crown rot, other species more tolerant of the site conditions (usually wetter) should replace them.  Since they share similar conditions conducive to infection, and since the symptoms of each of these diseases look very much alike, here are several web pages that will overlap a bit regarding the types of root and crown rot typically seen:Root, Crown, and Stem Rots on FlowersRhizoctonia Web Blight on FlowersOvergrown GroundcoversRoot Rots of Indoor Plants For now, you can try to trim away or comb out with your fingers any mushy growth under the live ends of the stems. Otherwise, you can focus more on monitoring the plants for water to make sure they're only watered when necessary and, for the future, consider amending the potting mix for improved drainage if growing species with greater moisture sensitivity.  Regarding the black, orange, or yellow spots on other garden plants, lawn, houseplants, and vines that you mention, these could be any number of infections, insect activity, normal leaf shed depending on the specific plant type and where on the foliage it's appearing. Please share photos of these so we can attempt a diagnosis. Fungicides cannot cure existing infections but might be warranted in a few cases to protect healthy growth from becoming infected. Sometimes, though, the fungal damage is only an eyesore and doesn't threaten the health of the plant enough to warrant treatment. (Or the opportunity for treatment has passed by the time symptoms appear.) If you have information about how some of those other plants are cared for (for example, is the lawn or garden irrigated using an automatic sprinkler system?), that can help us narrow-down possible causes of damage.  Miri
user: ﻿Thank you so m uh for the thorough response. In the barrel, I have lava stone, perilite, and a number of added measures to try and increase airflow but I’ll try and salvage what I can.   Here’s more pictures of elsewhere in the yard and inside the home. I’m really concerned about the pitch black stems in several of my plants. I’m pretty sure there’s mold in my grass (big white spots) then the trees are covered in vines and the vines have a ton of yellow, brown and black spots. Inside plants are also showing similar patterns (yellow spots that keep spreading). After I used fungicides I feel my plants look worse and now the healthy ones are covered in burns. I’ll add more pictures in next email.   Sent from my iPhone  On May 16, 2023, at 1:18 PM, Ask Extension <<personal data hidden>> wrote:    ﻿

Respond using exactly one of these formats:
Decision: clarify
Clarifying question: <one question needed before diagnosis or management>

Decision: respond
Answer: <concise agricultural answer>
```

**Target**

```text
Decision: clarify
Clarifying question: Could you provide close-up images of the symptoms on each type of plant, and share more details about the environmental conditions in your garden and home, such as humidity and temperature? Additionally, information on your watering and fertilization schedule would be helpful.
```

### clarify_or_respond.3 `mirage-#876931`

- Source: `mirage`
- Verifier: `clarify`
- Images: `data/raw/mirage/full/images/MMMT_Decomp/train/MMMT_Decomp-train-002036-01.png`, `data/raw/mirage/full/images/MMMT_Decomp/train/MMMT_Decomp-train-002036-02.png`, `data/raw/mirage/full/images/MMMT_Decomp/train/MMMT_Decomp-train-002036-03.png`

**Prompt**

```text
user: I have some purple coneflowers that either have mites or yellow asters.  There’s a lot of information online and it just confuses me.  I will include a few pics in case there’s anyone that can help.  I realized I didn’t take pictures of the plant itself, but they appear to be pretty healthy.  Thank you

Respond using exactly one of these formats:
Decision: clarify
Clarifying question: <one question needed before diagnosis or management>

Decision: respond
Answer: <concise agricultural answer>
```

**Target**

```text
Decision: clarify
Clarifying question: Could you describe any other symptoms you have observed on the coneflowers, such as leaf discoloration or stunted growth? Additionally, have there been any recent changes in environmental conditions, like weather or soil quality, that might be affecting the plants?
```

### clarify_or_respond.4 `mirage-#837616`

- Source: `mirage`
- Verifier: `clarify`
- Images: `data/raw/mirage/full/images/MMMT_Direct/train/MMMT_Direct-train-003621-01.png`, `data/raw/mirage/full/images/MMMT_Direct/train/MMMT_Direct-train-003621-02.png`, `data/raw/mirage/full/images/MMMT_Direct/train/MMMT_Direct-train-003621-03.png`

**Prompt**

```text
user: It appears that some of my maple tree leaves are deteriorating.  The range of the leaves go from healthy  to a few white spots along the spines to a lot to dying. The backside of a small number of leaves have white bugs.  The leaves are almost completely dead on three lower branches of the tree.  We have two additional maple trees with one having a similar problem on a much smaller scale, no dead leaves.

Thank you for helping me!!!

Respond using exactly one of these formats:
Decision: clarify
Clarifying question: <one question needed before diagnosis or management>

Decision: respond
Answer: <concise agricultural answer>
```

**Target**

```text
Decision: clarify
Clarifying question: Could you provide more details about the type of white bugs you are seeing on the leaves? Additionally, have you tried any treatments or noticed similar symptoms in other nearby plants or trees?
```

### clarify_or_respond.5 `mirage-#840279`

- Source: `mirage`
- Verifier: `clarify`
- Images: `data/raw/mirage/full/images/MMMT_Direct/train/MMMT_Direct-train-000312-01.png`

**Prompt**

```text
user: Hi, I have an old fashioned single peony plant that I moved from Iowa to Cleveland and now it is covered in a white film. Trying to avoid bad chemicals.

Respond using exactly one of these formats:
Decision: clarify
Clarifying question: <one question needed before diagnosis or management>

Decision: respond
Answer: <concise agricultural answer>
```

**Target**

```text
Decision: clarify
Clarifying question: Could you describe the texture and spread pattern of the white film on your peony? Additionally, can you provide details about the environmental conditions such as humidity and sunlight exposure, and let me know if you've tried any treatments so far?
```

### clarify_or_respond.6 `mirage-#841507`

- Source: `mirage`
- Verifier: `clarify`
- Images: `data/raw/mirage/full/images/MMMT_Decomp/train/MMMT_Decomp-train-001862-01.png`

**Prompt**

```text
user: Every year about this time this weed appears in my yard. With the lack of rain there are lots more of them this summer.  Would like to know what the name is and what I can spray them with to kill them but not harm my grass.
expert: Thank you for writing. I have lots of spotted spurge. It seems easy to pull out but leaves root fragments that regrow.  https://www.minnesotawildflowers.info/flower/prostrate-spurge#:~:text=The%20only%20other%20hairy%2C%20mat,surface%2C%20and%20seed%20lacks%20conspicuous
user: Thanks for your quick reply. Please confirm I can spray these pesky weeds with a typical post emergent herbicide. I have some Spectracide Weed Stop for lawns concentrate on hand and I think this will do the trick. On Wed, Jul 19, 2023 at 1:21 PM Ask Extension <<personal data hidden>> wrote:

Respond using exactly one of these formats:
Decision: clarify
Clarifying question: <one question needed before diagnosis or management>

Decision: respond
Answer: <concise agricultural answer>
```

**Target**

```text
Decision: respond
Answer: Spectracide Weed Stop for Lawns is a selective post-emergent herbicide that is generally effective against broadleaf weeds like spotted spurge. It should be safe for your grass if applied according to the label instructions. However, given the poor soil conditions and the fact that your grass goes dormant quickly, ensure that you apply it when the grass is actively growing to minimize stress. Also, consider spot-treating the spurge to reduce the risk of overspray. Since you plan to replace the soil and grass eventually, this approach should help manage the spurge in the meantime.
```

### clarify_or_respond.7 `mirage-#877669`

- Source: `mirage`
- Verifier: `clarify`
- Images: `data/raw/mirage/full/images/MMMT_Decomp/train/MMMT_Decomp-train-000778-01.png`, `data/raw/mirage/full/images/MMMT_Decomp/train/MMMT_Decomp-train-000778-02.png`, `data/raw/mirage/full/images/MMMT_Decomp/train/MMMT_Decomp-train-000778-03.png`

**Prompt**

```text
user: Please tell me what to do with the dead, curling leaves and spots on leaves?

Respond using exactly one of these formats:
Decision: clarify
Clarifying question: <one question needed before diagnosis or management>

Decision: respond
Answer: <concise agricultural answer>
```

**Target**

```text
Decision: clarify
Clarifying question: Could you please provide more details about the type of plant affected and any recent changes in its environment or care? Additionally, have you noticed any pests or other signs of disease?
```

### clarify_or_respond.8 `mirage-#827760`

- Source: `mirage`
- Verifier: `clarify`
- Images: `data/raw/mirage/full/images/MMMT_Direct/train/MMMT_Direct-train-001248-01.png`, `data/raw/mirage/full/images/MMMT_Direct/train/MMMT_Direct-train-001248-02.png`, `data/raw/mirage/full/images/MMMT_Direct/train/MMMT_Direct-train-001248-03.png`

**Prompt**

```text
user: We took in an emaciated blue heeler abandoned in the woods on the Big Island of Hawaii’i. I started getting 1x2 inch oval patterns of red dots on my arms and legs with 1/4' hairs sticking out at a 45 degree angle. They turned into deep oval wounds with white worms around the edges that I couldn't remove. Eventually I pooped out hundreds of creatures that looked like 2" long shrimp with long snouts. What could this be and did I get it from our dog?

Respond using exactly one of these formats:
Decision: clarify
Clarifying question: <one question needed before diagnosis or management>

Decision: respond
Answer: <concise agricultural answer>
```

**Target**

```text
Decision: clarify
Clarifying question: Can you provide more information about the dog's health and any symptoms it might be showing? Additionally, has the dog been examined by a veterinarian since you took it in?
```


## classification

### classification.1 `ip102-ip102-v1-1-images-42900-jpg`

- Source: `ip102`
- Verifier: `label`
- Images: `data/raw/ip102/full/ip102_v1.1/images/42900.jpg`

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
Answer: Colomerus vitis
Evidence: Visible agricultural symptoms or pest features support this label.
```

### classification.2 `plantvillage-train-030480`

- Source: `plantvillage`
- Verifier: `label`
- Images: `data/raw/plantvillage/full/images/train/030480.png`

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
Answer: tomato bacterial spot
Evidence: Visible Tomato symptoms support the Bacterial spot label.
```

### classification.3 `plantdoc-6390`

- Source: `plantdoc`
- Verifier: `label`
- Images: `data/raw/plantdoc/full/images/train/001764.png`

**Prompt**

```text
Identify the crop issue or pest in this agricultural image.

Choose exactly one label from this allowed label set:
Allowed labels: apple leaf; apple rust leaf; apple scab leaf; bell pepper leaf; bell pepper leaf spot; blueberry leaf; cherry leaf; corn gray leaf spot; corn leaf blight; corn rust leaf; grape leaf; grape leaf black rot; peach leaf; potato leaf early blight; potato leaf late blight; raspberry leaf; soyabean leaf; squash powdery mildew leaf; strawberry leaf; tomato early blight leaf; tomato leaf; tomato leaf bacterial spot; tomato leaf late blight; tomato leaf mosaic virus; tomato leaf yellow virus; tomato mold leaf; tomato septoria leaf spot; tomato two spotted spider mites leaf
Respond in this format:
Answer: <one allowed label>
Evidence: <brief visible symptom evidence>
Do not leave Answer blank, invent labels, or copy the placeholder text.
```

**Target**

```text
Answer: blueberry leaf
Evidence: Visible agricultural symptoms or pest features support this label.
```

### classification.4 `plantvillage-train-008180`

- Source: `plantvillage`
- Verifier: `label`
- Images: `data/raw/plantvillage/full/images/train/008180.png`

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
Answer: corn maize healthy
Evidence: Visible Corn (maize) symptoms support the healthy label.
```

### classification.5 `plantvillage-train-003394`

- Source: `plantvillage`
- Verifier: `label`
- Images: `data/raw/plantvillage/full/images/train/003394.png`

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

### classification.6 `ip102-ip102-v1-1-images-01693-jpg`

- Source: `ip102`
- Verifier: `label`
- Images: `data/raw/ip102/full/ip102_v1.1/images/01693.jpg`

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
Answer: paddy stem maggot
Evidence: Visible agricultural symptoms or pest features support this label.
```

### classification.7 `ip102-ip102-v1-1-images-23159-jpg`

- Source: `ip102`
- Verifier: `label`
- Images: `data/raw/ip102/full/ip102_v1.1/images/23159.jpg`

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
Answer: green bug
Evidence: Visible agricultural symptoms or pest features support this label.
```

### classification.8 `ip102-ip102-v1-1-images-16971-jpg`

- Source: `ip102`
- Verifier: `label`
- Images: `data/raw/ip102/full/ip102_v1.1/images/16971.jpg`

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


## consultation

### consultation.1 `agbase-agbase-634428`

- Source: `agbase`
- Verifier: `structured`
- Images: `data/raw/agbase/full/634428/634428_3.jpg`, `data/raw/agbase/full/634428/634428_2.jpg`, `data/raw/agbase/full/634428/634428_1.jpg`

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
Diagnosis: frost damage
Evidence: turning brown at edges and crumbling
Uncertainty: Image-only assessment; confirm with field context before treatment.
Management: keep an eye on the plants to see if new symptoms develop in healthy foliage as the weather gets warmer
Follow-up: Share close-up images and crop history if symptoms progress or the diagnosis is uncertain.
```

### consultation.2 `agbase-agbase-610071`

- Source: `agbase`
- Verifier: `structured`
- Images: `data/raw/agbase/full/610071/610071_1.jpg`

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
Diagnosis: tree, perennial plants
Evidence: Visible symptoms should be checked against the image and crop context.
Uncertainty: Image-only assessment; confirm with field context before treatment.
Management: No source-specific management step was provided; recommend local extension follow-up.
Follow-up: Share close-up images and crop history if symptoms progress or the diagnosis is uncertain.
```

### consultation.3 `agbase-agbase-630480`

- Source: `agbase`
- Verifier: `structured`
- Images: `data/raw/agbase/full/630480/630480_2.jpg`, `data/raw/agbase/full/630480/630480_1.jpg`

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
Diagnosis: northern watersnake
Evidence: Visible symptoms should be checked against the image and crop context.
Uncertainty: Image-only assessment; confirm with field context before treatment.
Management: No source-specific management step was provided; recommend local extension follow-up.
Follow-up: Share close-up images and crop history if symptoms progress or the diagnosis is uncertain.
```

### consultation.4 `agbase-agbase-656350`

- Source: `agbase`
- Verifier: `structured`
- Images: `data/raw/agbase/full/656350/656350_1.jpg`

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
Diagnosis: rose
Evidence: Visible symptoms should be checked against the image and crop context.
Uncertainty: Image-only assessment; confirm with field context before treatment.
Management: spray them if there are lots feeding on your roses
Follow-up: Share close-up images and crop history if symptoms progress or the diagnosis is uncertain.
```

### consultation.5 `agbase-agbase-655689`

- Source: `agbase`
- Verifier: `structured`
- Images: `data/raw/agbase/full/655689/655689_1.jpg`

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
Diagnosis: herbicide damage, likely 2,4-d
Evidence: Visible symptoms should be checked against the image and crop context.
Uncertainty: Image-only assessment; confirm with field context before treatment.
Management: No source-specific management step was provided; recommend local extension follow-up.
Follow-up: Share close-up images and crop history if symptoms progress or the diagnosis is uncertain.
```

### consultation.6 `agbase-agbase-870690`

- Source: `agbase`
- Verifier: `structured`
- Images: `data/raw/agbase/full/870690/870690_1.jpg`, `data/raw/agbase/full/870690/870690_2.jpg`

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
Diagnosis: ely pear tree
Evidence: Visible symptoms should be checked against the image and crop context.
Uncertainty: Image-only assessment; confirm with field context before treatment.
Management: No source-specific management step was provided; recommend local extension follow-up.
Follow-up: Share close-up images and crop history if symptoms progress or the diagnosis is uncertain.
```

### consultation.7 `agbase-agbase-656558`

- Source: `agbase`
- Verifier: `structured`
- Images: `data/raw/agbase/full/656558/656558_1.jpg`, `data/raw/agbase/full/656558/656558_3.jpg`

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
Diagnosis: pepper plant, tomato plant
Evidence: Visible symptoms should be checked against the image and crop context.
Uncertainty: Image-only assessment; confirm with field context before treatment.
Management: enclose plants in row cover material to prevent further damage
Follow-up: Share close-up images and crop history if symptoms progress or the diagnosis is uncertain.
```

### consultation.8 `agbase-agbase-796740`

- Source: `agbase`
- Verifier: `structured`
- Images: `data/raw/agbase/full/796740/796740_2.jpg`, `data/raw/agbase/full/796740/796740_3.jpg`, `data/raw/agbase/full/796740/796740_1.jpg`

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
Diagnosis: red ringspot virus
Evidence: reddish brown on the leaves
Uncertainty: Image-only assessment; confirm with field context before treatment.
Management: no treatments available, consider removing and replanting if plants are performing poorly, otherwise wait to see how the plants do in the next year or so
Follow-up: Share close-up images and crop history if symptoms progress or the diagnosis is uncertain.
```


## vqa

### vqa.1 `plantvillage_vqa-image_016499.JPG-048325`

- Source: `plantvillage_vqa`
- Verifier: `exact_match`
- Images: `data/raw/plantvillage_vqa/full/images/train/image_016499.JPG`

**Prompt**

```text
What is the medical/botanical diagnosis for these manifestations?

Respond in this format:
Answer: <short answer>
```

**Target**

```text
Answer: The leaf shows blotchy, asymmetrical yellowing, a classic symptom of HLB.
```

### vqa.2 `plantvillage_vqa-image_017823.JPG-052137`

- Source: `plantvillage_vqa`
- Verifier: `exact_match`
- Images: `data/raw/plantvillage_vqa/full/images/train/image_017823.JPG`

**Prompt**

```text
Assess this picture and provide a summarized botanical evaluation.

Respond in this format:
Answer: <short answer>
```

**Target**

```text
Answer: The cause is the bacterium Candidatus Liberibacter asiaticus, leading to Huanglongbing.
```

### vqa.3 `plantvillage_vqa-image_010779.JPG-031422`

- Source: `plantvillage_vqa`
- Verifier: `exact_match`
- Images: `data/raw/plantvillage_vqa/full/images/train/image_010779.JPG`

**Prompt**

```text
Can you find the source of this plant's poor condition?

Respond in this format:
Answer: <short answer>
```

**Target**

```text
Answer: Diagnosis: Leaf Mold.
```

### vqa.4 `plantvillage_vqa-image_043904.JPG-129095`

- Source: `plantvillage_vqa`
- Verifier: `exact_match`
- Images: `data/raw/plantvillage_vqa/full/images/train/image_043904.JPG`

**Prompt**

```text
Which disease-causing factor is behind these signs?

Respond in this format:
Answer: <short answer>
```

**Target**

```text
Answer: This is Bacterial Spot. Note the small, dark, water-soaked lesions.
```

### vqa.5 `plantvillage_vqa-image_033205.JPG-097741`

- Source: `plantvillage_vqa`
- Verifier: `exact_match`
- Images: `data/raw/plantvillage_vqa/full/images/train/image_033205.JPG`

**Prompt**

```text
What is the scientific identification of this plant sample?

Respond in this format:
Answer: <short answer>
```

**Target**

```text
Answer: Tomato
```

### vqa.6 `plantvillage_vqa-image_012346.JPG-036057`

- Source: `plantvillage_vqa`
- Verifier: `exact_match`
- Images: `data/raw/plantvillage_vqa/full/images/train/image_012346.JPG`

**Prompt**

```text
How would you evaluate the comprehensive wellness condition of this botanical sample?

Respond in this format:
Answer: <short answer>
```

**Target**

```text
Answer: This is a viral infection: TYLCV.
```

### vqa.7 `plantvillage_vqa-image_028777.JPG-084568`

- Source: `plantvillage_vqa`
- Verifier: `exact_match`
- Images: `data/raw/plantvillage_vqa/full/images/train/image_028777.JPG`

**Prompt**

```text
What accounts for the abnormalities in this leaf?

Respond in this format:
Answer: <short answer>
```

**Target**

```text
Answer: The plant is affected by Bacterial Spot disease.
```

### vqa.8 `plantvillage_vqa-image_018127.JPG-053022`

- Source: `plantvillage_vqa`
- Verifier: `exact_match`
- Images: `data/raw/plantvillage_vqa/full/images/train/image_018127.JPG`

**Prompt**

```text
What infectious agent, if any, is causing symptoms on this tomato leaf?

Respond in this format:
Answer: <short answer>
```

**Target**

```text
Answer: The causal agent is the Tomato Yellow Leaf Curl Virus.
```

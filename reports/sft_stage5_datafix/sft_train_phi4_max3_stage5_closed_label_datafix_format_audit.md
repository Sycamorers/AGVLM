# SFT Format Audit

- Manifest: `/blue/hmedeiros/qinruoyao/agvlm/data/manifests/full/sft_train_phi4_max3_stage5_closed_label_datafix.jsonl`
- Prompt format: `instructional`
- Target format: `instructional`
- Unique examples rendered: `128`
- Validation failures: `0`

## Counts

| Task | Manifest rows | Rendered unique examples |
| --- | ---: | ---: |
| clarify_or_respond | 6482 | 32 |
| classification | 61632 | 32 |
| consultation | 25000 | 32 |
| vqa | 50000 | 32 |

## clarify_or_respond

### clarify_or_respond.1 `mirage-#879142`

- Source: `mirage`
- Verifier: `clarify`
- Images: `data/raw/mirage/full/images/MMMT_Decomp/train/MMMT_Decomp-train-000418-01.png`, `data/raw/mirage/full/images/MMMT_Decomp/train/MMMT_Decomp-train-000418-02.png`

**Prompt**

```text
user: Are recycled rubbet tree "rings" or mats (48" diam.) a good idea around young trees, or are the trees better off simply with mulch?  The mats are porous.
expert: Thank you for writing. Please do not use this product.  It will raise the soil temperature higher than what the superficial roots and normal soil bacteria are used to.  Surround the tree with 4-6 inches of mulch (pine bark chips are fine) but keep the mulch at least 6 inches away from the tree bark. This is an oak and I cannot see the tree entering the ground. When did was it planted?  Who planted it? Send me a picture of the bottom foot of the tree after the rubber ring is gone.
user: I planted it in 2020; it looked like this:       ….and photo from today after removing rubber tree ring:        On Jul 28, 2024, at 5:36 PM, Ask Extension <<personal data hidden>> wrote:

Respond using exactly one of these formats:
Decision: clarify
Clarifying question: <one question needed before diagnosis or management>

Decision: respond
Answer: <concise agricultural answer>
```

**Target**

```text
Decision: respond
Answer: Thank you for providing the photos and additional information. Since the rubber tree ring has been removed, it's important to focus on proper mulching to support the health of your oak tree. Apply 4-6 inches of mulch, such as pine bark chips, around the base of the tree, ensuring that the mulch is kept at least 6 inches away from the tree trunk to prevent moisture buildup and potential rot. This will help maintain soil moisture, regulate temperature, and suppress weeds, promoting healthy growth for your oak tree.
```

### clarify_or_respond.2 `mirage-#842682`

- Source: `mirage`
- Verifier: `clarify`
- Images: `data/raw/mirage/full/images/MMMT_Decomp/train/MMMT_Decomp-train-000845-01.png`, `data/raw/mirage/full/images/MMMT_Decomp/train/MMMT_Decomp-train-000845-02.png`, `data/raw/mirage/full/images/MMMT_Decomp/train/MMMT_Decomp-train-000845-03.png`

**Prompt**

```text
user: My Hinoki Cypress was planted last fall and has done well in it's location.
However, almost 1 year later it has begun browning at the tips in various places around the tree. It has also had some (normal-seeming) dieback of internal foliage due to (I assume) not quite enough light. The browning seems to be accelerating despite attempts at solving the problem. Can you help me figure out what is wrong?

Light: It gets about 4 hours of direct sun in the afternoon and evening, and indirect for another 4ish hours in the morning.
Water: It is watered about once a week (or when the top few inches of soil are dry) for about 30 seconds with a harden hose sprayer. I have tried periods of more frequent watering, but it has not seemed to help.
Food: I have fertilized it twice this year (once in Spring and once about a week ago) with 18-6-12 pellet fertilizer
Treatment: I have also tried treating it for fungus (with a copper fungicide) and inserts (with BT)
expert: The scattered brown tips might be due to a soil moisture issue or to a minor pest like caterpillars of the native moth Arborvitae Leafminer. (They are named for their arborvitae preferred host plant, but they can also consume Chamaecyparis.) It sounds like  a sufficient level of direct sunlight (more would be ok, though, as they are tolerant of full sun) but the watering pattern may need to change to keep the plant hydrated without over-watering. Feel the soil about six inches deep next to the roots and water well only when it's becoming somewhat dry to the touch at that depth. The duration of water you're giving the plant now sounds too light, where it won't soak in deeply enough to give the roots much water. For a plant this size, you may need to apply a few gallons per watering session, which can be done with a watering can, bucket with holes in its base that can slowly drip water into the soil, or a soaker hose loosely coiled around the drip line of the shrub. (You can of course water by hand with the hose, too, but ideally should use a water breaker nozzle to break the water stream into a gentler "rain" and it may need to be applied for several minutes to get that amount of water into the root zone. One way to do this without having the excess run off the soil surface is to water lightly, move on to other plants, and then come back to the plant to water again, which gives each watering time to soak in.) Our Watering Trees and Shrubs page has more advice on how and when to water.  If Arborvitae Leafminer is present, they are not major pests and the simplest approach to their damage is to just carefully trim off those brown tips. Pesticide use would be a bit overkill but it's also hard to find a low-toxicity option that will impact any insect feeding inside of foliage as these do, where they are shielded from direct contact with a spray. Spinosad, a leaf-absorbed and naturally-derived ingredient, might work if the population were to boom in a future year. (Bt can work well for young caterpillars like bagworms, but in this scenario, since the caterpillars are inside foliage and Bt is not absorbed into the plant, it would be ineffective.) Arborvitae Leafminer has only one generation per year and eggs are laid around midsummer (July-ish) so if needed, an application made around this time of year or in August sometime may be successful in interrupting that generation. The pictured plant does not look severely infested, though, so for now we would not recommend use of an insecticide. Additionally, this kind of feeding damage can look nearly identical to water stress or root dieback injury to the foliage, so close examination would be needed to confirm the cause of browning was insects and not other factors. Leafminer insects emerge from their feeding chambers through a small hole in the foliage, so if none of the brown tips has a tiny hollow somewhere in that region of damage, leafminer would be less likely.  Hinoki Cypress grow slowly enough, and shrubs in general rarely suffer from nutrient deficiencies, that fertilization should not be necessary. Use caution if fertilizing as over-dosing can stress or injure roots, leading to similar-looking damage of yellowing foliage, brown tips, and a reduction in vigor. Discontinue use of the copper fungicide because a fungal infection is not indicated by the plant's symptoms. (Even though root rot organisms are fungus-like, they are not treatable with a copper-based product and not curable once they set in.) Over-applying copper can also damage plants and fungicides as a group sometimes risk harming other organisms, so always try to confirm the presence of a particular disease before using any sort of treatment. Fungicides are preventative measures and cannot cure existing disease nor reverse its damage.  Overall, the plant appears to be in decent shape, and adjusting your watering approach may improve its condition and tolerance for high heat. (Hinoki Cypress prefer the generally-cooler weather of the Pacific northwest -- where most of those in the nursery trade are grown -- and while they handle our heat admirably, it may tax their vigor a bit.)  Miri
user: Miri - thank you so much for your thoughtful and in-depth response! It sounds like water is the likely culprit - so I'll pay close attention there. I will also cut the other treatments noting your valuable comments there. Thanks again!Best,--DwightOn Thu, Jul 27, 2023 at 10:33 AM Ask Extension <<personal data hidden>> wrote:

Respond using exactly one of these formats:
Decision: clarify
Clarifying question: <one question needed before diagnosis or management>

Decision: respond
Answer: <concise agricultural answer>
```

**Target**

```text
Decision: clarify
Clarifying question: Could you provide more information about the soil type and drainage conditions? Additionally, have you noticed any visible pests or signs of disease on the plant? Any recent changes in the environment or nearby plants that might affect the Hinoki Cypress? Also, could you describe what you see in the attached images?
```

### clarify_or_respond.3 `mirage-#840068`

- Source: `mirage`
- Verifier: `clarify`
- Images: `data/raw/mirage/full/images/MMMT_Decomp/train/MMMT_Decomp-train-001165-01.png`

**Prompt**

```text
user: This guy is eating my dahlia. Can you tell me what it is and how to get rid of it without using toxic sprays.

Respond using exactly one of these formats:
Decision: clarify
Clarifying question: <one question needed before diagnosis or management>

Decision: respond
Answer: <concise agricultural answer>
```

**Target**

```text
Decision: clarify
Clarifying question: Could you provide a description of the pest or any details about the damage it is causing? Additionally, have you tried any methods to control it so far?
```

### clarify_or_respond.4 `mirage-#880875`

- Source: `mirage`
- Verifier: `clarify`
- Images: `data/raw/mirage/full/images/MMMT_Decomp/train/MMMT_Decomp-train-000023-01.png`, `data/raw/mirage/full/images/MMMT_Decomp/train/MMMT_Decomp-train-000023-02.png`, `data/raw/mirage/full/images/MMMT_Decomp/train/MMMT_Decomp-train-000023-03.png`

**Prompt**

```text
user: A number of perennials are displaying signs of distress-stems bending down, newer leaves on tops of plants are smaller, blooms are deformed, newer buds on black-eyed Susans are brown and appear to be dead.  Affected plants include:
goldenrods, coneflowers, cardinal flower, phlox, black-eyed Susans
expert: If there are no other signs of pest or disease damage, the plants are probably suffering from heat stress. Plants are sensitive to both air and soil surface temperatures. Temperatures ranging from 60° to 85°F are optimum for plant growth and development.   Plants have limited ability to regulate temperature and begin to experience stress as air temperatures rise above 86°F.   Water evaporation through pores in the leaves is the primary cooling mechanism used by plants. If a plant doesn’t have enough water, the pores in the leaves close and the plant can no longer keep itself cool. When temperatures are exceptionally high and prolonged, plants can lose water through evaporation much faster than the roots can absorb it.  Wilted, drooping leaves, resulting from a combination of excess water loss and high internal temperatures, are often the first signs of heat stress.   Sunburn is another type of heat damage caused by excessive exposure to direct sunlight. Sunburned leaves and stems become discolored and appear scorched.  Plants experiencing continued heat stress will be unable to maintain the processes necessary for growth and development and will become stunted and vulnerable to insect attack.  So what can be done to minimize effects of heat stress:  Water thoroughly in the morning on days when high temperatures are expected, to help plants cope with excess heat. In general, when watering plants, do it deeply and infrequently so that they put their energy into sending roots down into the lower depths of the soil. During a period of drought, the roots are then able to access moisture reserves deeper in the soil.  If plants are watered too lightly, then the moisture only penetrates the top layer of soil, resulting in plants with shallow root systems.  During a heatwave, the water in the top inch of soil quickly evaporates, so heat-stressed plants also need deep watering.If discovered in time during a heat wave, a drooping wilted plant can often be revived with additional water.   Watch for radiant heat - Pay special attention to plants next to walkways, patios, and other structures that radiate heat. They may be the first to show signs of heat stress and the need for extra water.  Add mulch - Apply two to four inches of organic mulch over the root areas around garden plants, trees, and shrubs. This will lower soil temperature and diminish moisture loss.  Provide shade - Use shade cloth to provide direct sun protection and reduce the air temperature around sun and heat-exposed plants.   Avoid applying fertilizers during extreme heat as they stimulate growth and add to plant stress. Fertilizers also absorb moisture from the root zone.  Many plants will stop blooming during extreme heat. Removing the spent flowers can improve the plant's appearance, keep plants from spending precious resources on fruit and seed development, and set the plant up to bloom again once the heat wave passes.  Following is a link to Iowa State Univ Extension with more info about managing the garden in a heat wave: https://yardandgarden.extension.iastate.edu/how-to/managing-garden-extreme-heat
user: Thanks for your prompt response. I'm sorry I provided you with very skimpy information.  These plants and the damage are confined to one area; other areas of unaffected so far.  We do follow the watering tips and do not fertilize the native pollinators.  Does this point to a fungal disease?  How can I identify whether or not it is a fungal disease?  If it is, what are the treatment options?   Thank you!
expert: If you think it might be a fungal disease it would be helpful if you could send some close-up pictures of leaves that might be spotted or discolored as well as pictures of deformed flowers. The close-ups would help in identifying problem other than heat wilt. Following are a few links to University Extension sites that provide more info on common fungal diseases with pictures that may help you determine if your plants are affected. Many if not most fungal diseases show up on the leaves. https://www.extension.iastate.edu/news/yard-and-garden-common-diseases-perennials https://extension.umd.edu/resource/fungal-leaf-spot-disease-flowers/ https://extension.psu.edu/key-herbaceous-plants-and-key-pathogens-diseases
user: Thank you!  This is very helpful.  I will send some photos tomorrow (Sunday).   On Sat, Aug 10, 2024 at 1:42 PM Ask Extension <<personal data hidden>> wrote:

Respond using exactly one of these formats:
Decision: clarify
Clarifying question: <one question needed before diagnosis or management>

Decision: respond
Answer: <concise agricultural answer>
```

**Target**

```text
Decision: clarify
Clarifying question: Could you please describe any specific symptoms visible in the photos you sent, such as leaf spots, discoloration, or any other notable features? This will help in identifying if a fungal disease is present.
```

### clarify_or_respond.5 `mirage-#879732`

- Source: `mirage`
- Verifier: `clarify`
- Images: `data/raw/mirage/full/images/MMMT_Decomp/train/MMMT_Decomp-train-002410-01.png`

**Prompt**

```text
user: Please help. Our Italian plum is not doing well this year after a great season last year.

Respond using exactly one of these formats:
Decision: clarify
Clarifying question: <one question needed before diagnosis or management>

Decision: respond
Answer: <concise agricultural answer>
```

**Target**

```text
Decision: clarify
Clarifying question: Could you describe any specific symptoms the tree is showing, such as leaf discoloration, fruit drop, or signs of pests? Additionally, have there been any changes in care, environment, or weather conditions this year compared to last?
```

### clarify_or_respond.6 `mirage-#826609`

- Source: `mirage`
- Verifier: `clarify`
- Images: `data/raw/mirage/full/images/MMMT_Direct/train/MMMT_Direct-train-002817-01.png`

**Prompt**

```text
user: Hello. For several years, a portion of our lawn starts out the spring practically white in color, and then it greens up over the summer (see attached photo). I think the affected area has grown a bit each year. Some lawn services have suggested we have grubs, and we did get one treatment last spring, but it didn't help. What do you think it is and how should I treat it? Thank you!

Respond using exactly one of these formats:
Decision: clarify
Clarifying question: <one question needed before diagnosis or management>

Decision: respond
Answer: <concise agricultural answer>
```

**Target**

```text
Decision: respond
Answer: Based on the information provided, it seems unlikely that grubs are the cause of the white discoloration, especially since the lawn service has been maintaining the lawn well and there are no signs of bird activity. The symptoms you describe, with the grass appearing white and dried up, could be indicative of a fungal issue such as snow mold, which can occur when snow covers the grass for extended periods. Snow mold typically resolves as temperatures rise and the lawn dries out, which aligns with your observation of the lawn greening up over the summer.

To address this, consider the following steps:
1. Rake the affected area gently in early spring to help the grass dry out and to remove any matted grass.
2. Ensure good air circulation by keeping the lawn mowed at the recommended height and removing any debris.
3. Avoid excessive nitrogen fertilization in late fall, as this can promote snow mold.
4. If snow mold is a recurring issue, you might consider applying a fungicide in late fall before the first snowfall.

Taking a soil sample for analysis could also be beneficial to rule out any nutrient deficiencies or pH imbalances that might be contributing to the problem. This will provide a clearer picture of the soil health and help tailor any further treatments. If the problem persists despite these measures, consulting with a local extension service or a turf specialist might provide additional insights specific to your region.
```

### clarify_or_respond.7 `mirage-#883937`

- Source: `mirage`
- Verifier: `clarify`
- Images: `data/raw/mirage/full/images/MMMT_Decomp/train/MMMT_Decomp-train-000485-01.png`, `data/raw/mirage/full/images/MMMT_Decomp/train/MMMT_Decomp-train-000485-02.png`

**Prompt**

```text
user: We have a honeycrisp apple tree in our backyard in Littleton. This year is the second year we've gotten a nice crop of apples. A few years ago, when we had apples, they were small but beautiful and delicious. This year, the apples are much bigger, but they have lots of dark brown spots on them. Even the apples that are still green have darker green spots showing up. What is causing this? Are the apples still good to eat? And how do we prevent this in the future?

Respond using exactly one of these formats:
Decision: clarify
Clarifying question: <one question needed before diagnosis or management>

Decision: respond
Answer: <concise agricultural answer>
```

**Target**

```text
Decision: respond
Answer: The symptoms you are describing, such as the dark brown spots and the darker green spots on the apples, are consistent with a condition known as 'bitter pit.' Bitter pit is a physiological disorder caused by a calcium deficiency in the fruit. This can be exacerbated by factors such as aggressive pruning and rapid growth, which you mentioned occurred this year.

The apples with bitter pit are generally safe to eat, although the affected areas may be bitter and unpleasant in taste. You can cut out the affected parts and consume the rest of the apple if it appears otherwise healthy.

To prevent bitter pit in the future, consider the following steps:
1. Ensure consistent watering, especially during dry spells, to help the tree uptake calcium more effectively.
2. Avoid excessive pruning, as this can lead to rapid growth that exacerbates calcium deficiency.
3. Consider applying a calcium spray to the developing fruit during the growing season to help mitigate the issue.
4. Conduct a soil test to check for nutrient imbalances and amend the soil as necessary.

These steps should help reduce the occurrence of bitter pit in future harvests.
```

### clarify_or_respond.8 `mirage-#874644`

- Source: `mirage`
- Verifier: `clarify`
- Images: `data/raw/mirage/full/images/MMMT_Decomp/train/MMMT_Decomp-train-003344-01.png`, `data/raw/mirage/full/images/MMMT_Decomp/train/MMMT_Decomp-train-003344-02.png`, `data/raw/mirage/full/images/MMMT_Decomp/train/MMMT_Decomp-train-003344-03.png`

**Prompt**

```text
user: Our bushes have webs that started in the boxwoods and then spread to others.  Last year we used the hose to spray the webs, which helped, but this year the webs expanded to more territory. We'd like to know what is causing the webs and how to eliminate them permanently.  Thank you.

Respond using exactly one of these formats:
Decision: clarify
Clarifying question: <one question needed before diagnosis or management>

Decision: respond
Answer: <concise agricultural answer>
```

**Target**

```text
Decision: respond
Answer: The symptoms you describe, with webs binding leaves and causing them to turn brown, are indicative of an infestation by spider mites or possibly webworms. Spider mites are tiny pests that can cause significant damage to plants by sucking the sap from the leaves, leading to browning and leaf drop. Webworms, on the other hand, create webs to protect themselves while they feed on the leaves.

To address this issue, you can start by using a strong jet of water to dislodge the webs and pests, as you did last year. However, for a more permanent solution, consider using insecticidal soap or neem oil, which are effective against spider mites and are safe for most plants. Apply these treatments according to the product instructions, ensuring thorough coverage of the affected areas. Additionally, maintaining good plant health through proper watering and fertilization can help your plants resist pest infestations.

If the problem persists, you may want to consult a local extension service or a professional pest control service for further assistance.
```

### clarify_or_respond.9 `mirage-#830730`

- Source: `mirage`
- Verifier: `clarify`
- Images: `data/raw/mirage/full/images/MMMT_Decomp/train/MMMT_Decomp-train-002684-01.png`, `data/raw/mirage/full/images/MMMT_Decomp/train/MMMT_Decomp-train-002684-02.png`

**Prompt**

```text
user: I planted 10 Campfire roses the summer of 2020 and enjoyed them for 3 summers.  They didn't start growing this year, and they appear to be dead.  When I tug on the branches, they feel as if there are no roots keeping them in the ground.  There are piles of dirt granules on top of the leaves that gathered around the base of each plant over the winter.  I am wondering if something has eaten the roots.  I don't want to plant new ones until I know what might have happened and how to prevent.  I did have japanese beetles on them last summer and hand picked them into soapy hot water diligently.  I am attempting to attach two images.  Do you have any advice?

Respond using exactly one of these formats:
Decision: clarify
Clarifying question: <one question needed before diagnosis or management>

Decision: respond
Answer: <concise agricultural answer>
```

**Target**

```text
Decision: respond
Answer: Based on the information provided, it seems likely that your Campfire roses have been affected by root-feeding pests, possibly root weevils or nematodes, which could explain the lack of root anchorage and the presence of dirt granules. The mustard test revealing worms suggests the presence of soil-dwelling pests. To prevent this issue in the future, consider the following steps:

1. **Soil Treatment:** Before replanting, treat the soil to eliminate any remaining pests. You can use beneficial nematodes, which are natural predators of many soil-dwelling pests, or consider a soil drench with an appropriate pesticide if the infestation is severe.

2. **Soil Health:** Improve soil health by adding organic matter, such as compost, to encourage beneficial organisms that can outcompete harmful pests.

3. **Regular Monitoring:** Keep an eye on your plants for any signs of pest activity. Early detection can help manage infestations before they become severe.

4. **Companion Planting:** Consider planting companion plants that repel pests, such as marigolds, which are known to deter nematodes.

5. **Japanese Beetle Management:** Continue your diligent management of Japanese beetles, as they can weaken plants and make them more susceptible to other issues.

By addressing the soil health and monitoring for pests, you can create a more resilient environment for your new roses. If you have any further questions or need more specific advice, feel free to ask!
```

### clarify_or_respond.10 `mirage-#879105`

- Source: `mirage`
- Verifier: `clarify`
- Images: `data/raw/mirage/full/images/MMMT_Decomp/train/MMMT_Decomp-train-000161-01.png`, `data/raw/mirage/full/images/MMMT_Decomp/train/MMMT_Decomp-train-000161-02.png`

**Prompt**

```text
user: Hello experts, this is a magnolia tree. I found white spots on the branches, some branches are black, and sometimes I can see some flying insects on the tree. Some liquid will fall from the tree to the ground. What kind of pests and diseases may have occurred? How to deal with it? What kind of medicine should be sprayed?

Respond using exactly one of these formats:
Decision: clarify
Clarifying question: <one question needed before diagnosis or management>

Decision: respond
Answer: <concise agricultural answer>
```

**Target**

```text
Decision: clarify
Clarifying question: Could you provide more details about the appearance of the white spots and the black branches? Additionally, do you know what type of flying insects are present, or can you describe them? This information, along with any previous treatments you've tried, will help in diagnosing the issue more accurately.
```

### clarify_or_respond.11 `mirage-#825262`

- Source: `mirage`
- Verifier: `clarify`
- Images: `data/raw/mirage/full/images/MMMT_Direct/train/MMMT_Direct-train-000396-01.png`, `data/raw/mirage/full/images/MMMT_Direct/train/MMMT_Direct-train-000396-02.png`, `data/raw/mirage/full/images/MMMT_Direct/train/MMMT_Direct-train-000396-03.png`

**Prompt**

```text
user: As the snow has melted I noticed a couple of days ago this unusual marking in my lawn under a tree and extending into the lawn that appeared and each day since more have appeared. I thought when I first saw the marking it looked like someone had drug a branch across the ground. It is varies in depth but is generally 1/2  - 3/4 inch deep…any idea what this may be and if we need to be treating it to prevent further spread.

Respond using exactly one of these formats:
Decision: clarify
Clarifying question: <one question needed before diagnosis or management>

Decision: respond
Answer: <concise agricultural answer>
```

**Target**

```text
Decision: respond
Answer: It seems like you have already received the information you needed to identify the markings and how to address them. If you have any further questions or need additional assistance, feel free to reach out. I'm glad the provided resources were helpful!
```

### clarify_or_respond.12 `mirage-#837196`

- Source: `mirage`
- Verifier: `clarify`
- Images: `data/raw/mirage/full/images/MMMT_Decomp/train/MMMT_Decomp-train-001052-01.png`

**Prompt**

```text
user: Hello

 I have  2 Colossal Chestnut (Castanea sativa x crenata), 2 - Precoce Migoule Chestnut (Castanea sativa x crenata), and 4 - Chinese Chestnut (Castanea mollissima). The leaves of the Colossal trees look like they are being eaten by some type of insect. The leaves on the other trees look fine. Do you have any thoughts on what is eating the leaves and how would I control it? Both of these trees look like they will blossom for the first time this year. Whatever is eating these leaves looks similar to what eats the leaves of one of our rose bushes. Any insight you may have is greatly appreciated.

Respond using exactly one of these formats:
Decision: clarify
Clarifying question: <one question needed before diagnosis or management>

Decision: respond
Answer: <concise agricultural answer>
```

**Target**

```text
Decision: respond
Answer: Based on your description and the identification of rose chafers as the likely culprits, here are some steps you can take to manage and control them:

1. **Manual Removal**: If the infestation is not too severe, you can manually remove the rose chafers from the trees. This is best done in the early morning or late evening when they are less active.

2. **Neem Oil**: Applying neem oil can help deter rose chafers. It acts as a repellent and can disrupt their feeding and reproductive cycles. Follow the instructions on the product label for the best results.

3. **Insecticidal Soap**: This can be used to target the larvae and adult beetles. Ensure thorough coverage of the leaves.

4. **Beneficial Nematodes**: These can be applied to the soil to target the larvae stage of the rose chafers, reducing their population over time.

5. **Pheromone Traps**: These can be used to capture adult beetles, but be cautious as they might attract more beetles to your area.

6. **Regular Monitoring**: Keep an eye on your trees regularly to catch any new infestations early.

For the pin oak, if the damage is extensive and you suspect potato leafhoppers, you might consider consulting a local arborist for a more targeted approach, as they can provide specific treatments that might be more effective.

Implementing these measures should help in managing the rose chafers on your chestnut trees. If the problem persists, consider reaching out to a local extension service for further assistance.
```

### clarify_or_respond.13 `mirage-#885942`

- Source: `mirage`
- Verifier: `clarify`
- Images: `data/raw/mirage/full/images/MMMT_Direct/train/MMMT_Direct-train-000893-01.png`, `data/raw/mirage/full/images/MMMT_Direct/train/MMMT_Direct-train-000893-02.png`, `data/raw/mirage/full/images/MMMT_Direct/train/MMMT_Direct-train-000893-03.png`

**Prompt**

```text
user: I have an issue in the yard of the home we bought in May 2023. Over the last year we have personally installed new pollinator flower beds to cover much of the grassy areas. Since late June, about 75% of the yard has been taken over by what seems to be a fungal disease and I can't control with neem or copper fungicides. I will attach additional images once my post is approved.
user: Additional images are below:
user:

Respond using exactly one of these formats:
Decision: clarify
Clarifying question: <one question needed before diagnosis or management>

Decision: respond
Answer: <concise agricultural answer>
```

**Target**

```text
Decision: clarify
Clarifying question: Could you describe the specific symptoms you are observing on the plants, and let me know which types of plants are most affected? Additionally, any details about the environmental conditions, such as humidity levels, shade, and your watering schedule, would be helpful. Also, could you provide a brief description of what is shown in the attached images?
```

### clarify_or_respond.14 `mirage-#883372`

- Source: `mirage`
- Verifier: `clarify`
- Images: `data/raw/mirage/full/images/MMMT_Decomp/train/MMMT_Decomp-train-002536-01.png`, `data/raw/mirage/full/images/MMMT_Decomp/train/MMMT_Decomp-train-002536-02.png`

**Prompt**

```text
user: The poplar tree in our yard has 1/4 of its branches with no leaves and has dropped yellow leaves for the past 3 weeks.

Respond using exactly one of these formats:
Decision: clarify
Clarifying question: <one question needed before diagnosis or management>

Decision: respond
Answer: <concise agricultural answer>
```

**Target**

```text
Decision: clarify
Clarifying question: Could you provide more information about the location and climate conditions where the tree is planted, as well as any recent changes in its environment or care routine? Additionally, have you noticed any signs of pests or diseases on the tree?
```

### clarify_or_respond.15 `mirage-#832527`

- Source: `mirage`
- Verifier: `clarify`
- Images: `data/raw/mirage/full/images/MMMT_Decomp/train/MMMT_Decomp-train-000918-01.png`, `data/raw/mirage/full/images/MMMT_Decomp/train/MMMT_Decomp-train-000918-02.png`, `data/raw/mirage/full/images/MMMT_Decomp/train/MMMT_Decomp-train-000918-03.png`

**Prompt**

```text
user: Hello - the mimosa tree in my front yard is struggling! It was very slow to blossom this year and there are multiple limbs that aren’t blossoming / appear to be dead. Should we be concerned? Any suggestions as to how we can save this tree?

Respond using exactly one of these formats:
Decision: clarify
Clarifying question: <one question needed before diagnosis or management>

Decision: respond
Answer: <concise agricultural answer>
```

**Target**

```text
Decision: clarify
Clarifying question: Could you provide more information about the age of the tree, recent weather conditions, your watering and fertilization practices, and whether you've noticed any signs of pests or diseases? Additionally, any details about the soil conditions would be helpful.
```

### clarify_or_respond.16 `mirage-#877738`

- Source: `mirage`
- Verifier: `clarify`
- Images: `data/raw/mirage/full/images/MMMT_Direct/train/MMMT_Direct-train-000683-01.png`

**Prompt**

```text
user: Our dogs will eat these seeds like candy if we let them. Can you tell us what they are?

Respond using exactly one of these formats:
Decision: clarify
Clarifying question: <one question needed before diagnosis or management>

Decision: respond
Answer: <concise agricultural answer>
```

**Target**

```text
Decision: clarify
Clarifying question: Could you provide more details about the environment where this plant is growing, such as the region or specific conditions? Additionally, do you have any guesses or prior knowledge about what this plant might be?
```

### clarify_or_respond.17 `mirage-#828676`

- Source: `mirage`
- Verifier: `clarify`
- Images: `data/raw/mirage/full/images/MMMT_Direct/train/MMMT_Direct-train-003581-01.png`, `data/raw/mirage/full/images/MMMT_Direct/train/MMMT_Direct-train-003581-02.png`, `data/raw/mirage/full/images/MMMT_Direct/train/MMMT_Direct-train-003581-03.png`

**Prompt**

```text
user: Could you ID this bee collecting pollen from a patch of Baby Blue Eyes (Nemophila menziesii) in my Eugene backyard on 5-5-23?
It was about 1.3 cm long, and flew quickly from flower to flower so I had a hard time getting it into focus.  It has been visiting this patch of flowers for several days.

Respond using exactly one of these formats:
Decision: clarify
Clarifying question: <one question needed before diagnosis or management>

Decision: respond
Answer: <concise agricultural answer>
```

**Target**

```text
Decision: respond
Answer: Based on the details you've provided and your agreement with the identification, it seems likely that the bee you observed is indeed a black-tailed bumblebee. These bees are known for their quick movements and are common visitors to gardens with native flowers. Your efforts to transform your yard into a habitat for bees, butterflies, and birds are commendable and will likely attract a variety of pollinators. Keep observing, and you might spot even more species!
```

### clarify_or_respond.18 `mirage-#886078`

- Source: `mirage`
- Verifier: `clarify`
- Images: `data/raw/mirage/full/images/MMMT_Direct/train/MMMT_Direct-train-003700-01.png`, `data/raw/mirage/full/images/MMMT_Direct/train/MMMT_Direct-train-003700-02.png`

**Prompt**

```text
user: We planted some bell peppers in a raised bed and it seems like they turned the soil all around them a chalky color. Could you please tell us if that soil is now not useable and what the issue could be? The first picture is of the aforementioned soil and the second of the soil elsewhere in the bed.

Respond using exactly one of these formats:
Decision: clarify
Clarifying question: <one question needed before diagnosis or management>

Decision: respond
Answer: <concise agricultural answer>
```

**Target**

```text
Decision: clarify
Clarifying question: Could you provide more information on any recent soil amendments or treatments you have applied, as well as your watering and fertilization practices? Additionally, have you noticed any pests or diseases affecting the plants?
```

### clarify_or_respond.19 `mirage-#886097`

- Source: `mirage`
- Verifier: `clarify`
- Images: `data/raw/mirage/full/images/MMMT_Decomp/train/MMMT_Decomp-train-002041-01.png`, `data/raw/mirage/full/images/MMMT_Decomp/train/MMMT_Decomp-train-002041-02.png`

**Prompt**

```text
user: The crown of my potted brugmancia is becoming more gnarly and has splits in the wood and dead/loose bark. Crown rot? I'd hate to discard the plant, it is 8 ft tall and in full bloom.

Respond using exactly one of these formats:
Decision: clarify
Clarifying question: <one question needed before diagnosis or management>

Decision: respond
Answer: <concise agricultural answer>
```

**Target**

```text
Decision: respond
Answer: Based on the symptoms you've described, it does sound like your brugmansia might be suffering from crown rot, especially given the gnarly crown, splits in the wood, and dead/loose bark. Crown rot is often caused by fungal pathogens that thrive in overly moist conditions, which can be exacerbated by poor drainage or overwatering. Since your plant is in a pot and has been knocked over several times, it's possible that the root system has been stressed, which can contribute to these issues.

To address this, you can try the following steps:
1. **Improve Drainage:** Ensure that the pot has adequate drainage holes and that water is not pooling at the bottom. Consider using a well-draining potting mix.
2. **Reduce Watering:** Allow the top inch of soil to dry out between waterings to prevent excess moisture.
3. **Treat with Fungicide:** Apply a fungicide that is effective against crown rot to the affected areas, following the manufacturer's instructions.
4. **Prune Affected Areas:** Carefully prune away any dead or severely affected wood to prevent the spread of the rot.

Regarding taking cuttings, if the plant is indeed infected with a pathogen, there is a risk that the cuttings could carry the pathogen. However, if you take cuttings from healthy, unaffected parts of the plant and treat them with a fungicide, you may be able to propagate new, healthy plants. Ensure that your cutting tools are sterilized to prevent the spread of any pathogens.

Given the size and age of your plant, it might be worth trying to save it while also taking a few cuttings as a backup. Good luck, and I hope your brugmansia recovers soon!
```

### clarify_or_respond.20 `mirage-#881421`

- Source: `mirage`
- Verifier: `clarify`
- Images: `data/raw/mirage/full/images/MMMT_Decomp/train/MMMT_Decomp-train-003140-01.png`

**Prompt**

```text
user: My Linden is 4 years old and get sfull intense  sun the leaves are turning brown and also has holes in the leaves.  Does that sound like scorch and aphids?  I am not sure how to treat that other than more water?

Respond using exactly one of these formats:
Decision: clarify
Clarifying question: <one question needed before diagnosis or management>

Decision: respond
Answer: <concise agricultural answer>
```

**Target**

```text
Decision: respond
Answer: Based on the symptoms you've described, it does sound like your Linden tree might be experiencing leaf scorch due to intense sun exposure, as well as potential pest damage, possibly from aphids or beetles. Leaf scorch can occur when the tree is unable to take up enough water to compensate for the water lost through transpiration, especially in full sun. Ensure the tree is receiving adequate water, particularly during hot, dry periods. For the holes in the leaves, inspect the tree for signs of aphids or beetles. Aphids can often be managed with a strong spray of water to dislodge them or with insecticidal soap. If beetles are present, you may need to identify the specific type to determine the best control method. Regular monitoring and maintaining tree health through proper watering and mulching can help prevent further issues.
```

### clarify_or_respond.21 `mirage-#826985`

- Source: `mirage`
- Verifier: `clarify`
- Images: `data/raw/mirage/full/images/MMMT_Direct/train/MMMT_Direct-train-002845-01.png`

**Prompt**

```text
user: I’ve noticed the top layer of bark splitting open on branches of a 3 year old (otherwise healthy) viburnum close to the bottom . The entire branch then dues. I’m uploading a picture of the most affected branches but it’s also happening on smaller branches.

Respond using exactly one of these formats:
Decision: clarify
Clarifying question: <one question needed before diagnosis or management>

Decision: respond
Answer: <concise agricultural answer>
```

**Target**

```text
Decision: clarify
Clarifying question: Can you provide more information about the environmental conditions around your viburnum, such as recent weather patterns, watering habits, and any signs of pests or disease? Additionally, have there been any recent changes or treatments applied to the plant?
```

### clarify_or_respond.22 `mirage-#872214`

- Source: `mirage`
- Verifier: `clarify`
- Images: `data/raw/mirage/full/images/MMMT_Direct/train/MMMT_Direct-train-000523-01.png`

**Prompt**

```text
user: Please see the pic of a maple leaf.   Yes, we have plenty of spongy moth caterpillars, but none of the other plants they have attacked have these spots on the leaves.  Does this tree have a disease/infestation in addition to the caterpillars and if so, how should we proceed?

Thank you.
expert: Barbara, Looks like tar spot.  You can apply a fungicide making sure the entire leaf is covered.  As an ongoing measure, rake up fallen leaves and destroy them.  In the Fall makes sure all leaves are raked, removed and destroyed.  The fungus does survive throughout the winter, overwinter.  Attached is a fact sheet. tarspotofmaple.pdf  Good luck, Cynthia B, MGV
user: Thanks. This is very helpful.    the pdf is not clickable.     From: ask=<personal data hidden> <ask=<personal data hidden>> On Behalf Of Ask Extension Sent: Monday, June 10, 2024 11:43 AM To: Barbara Lawrence <<personal data hidden>> Subject: Re: caterpillars or disease? (#0142132)
expert: http://plantclinic.cornell.edu/factsheets/tarspotofmaple.pdf
user: Thank you... I have done a bit of googling since you told me what it is.    This link is broken at the Cornell end it appears.    From: ask=<personal data hidden> <ask=<personal data hidden>> On Behalf Of Ask Extension Sent: Monday, June 10, 2024 11:57 AM To: Barbara Lawrence <<personal data hidden>> Subject: Re: caterpillars or disease? (#0142132)

Respond using exactly one of these formats:
Decision: clarify
Clarifying question: <one question needed before diagnosis or management>

Decision: respond
Answer: <concise agricultural answer>
```

**Target**

```text
Decision: respond
Answer: It seems like you've already identified the issue as tar spot, which is a fungal disease, and received advice on how to manage it. If the link to the fact sheet is broken, I recommend checking other reliable sources such as university extension websites or contacting a local extension office for more information. If you have any further questions or need additional assistance, feel free to ask!
```

### clarify_or_respond.23 `mirage-#879441`

- Source: `mirage`
- Verifier: `clarify`
- Images: `data/raw/mirage/full/images/MMMT_Direct/train/MMMT_Direct-train-000520-01.png`, `data/raw/mirage/full/images/MMMT_Direct/train/MMMT_Direct-train-000520-02.png`

**Prompt**

```text
user: Hello, Cameron you tell me what is causing the yellowing of my grass and how do I correct it, please?  Thank you. Bob

Respond using exactly one of these formats:
Decision: clarify
Clarifying question: <one question needed before diagnosis or management>

Decision: respond
Answer: <concise agricultural answer>
```

**Target**

```text
Decision: clarify
Clarifying question: Could you provide more details about your lawn, such as the type of grass, recent weather conditions, your watering and fertilization schedule, and whether you've noticed any pests or diseases? This information will help in diagnosing the issue.
```

### clarify_or_respond.24 `mirage-#823321`

- Source: `mirage`
- Verifier: `clarify`
- Images: `data/raw/mirage/full/images/MMMT_Direct/train/MMMT_Direct-train-003106-01.png`

**Prompt**

```text
user: This has been living on a windowsill in partial sun. It is lanky and in a 5x9 inch pot. I wish to repot it. What type of soil, pot and can it be trimmed back? If able to trim, how far?
The area with all the leaves is lovely. Can the stems be put in water or soil and create another plant? This may not be all the questions one should be asking.

Respond using exactly one of these formats:
Decision: clarify
Clarifying question: <one question needed before diagnosis or management>

Decision: respond
Answer: <concise agricultural answer>
```

**Target**

```text
Decision: clarify
Clarifying question: Could you please specify the type of plant you are referring to? This will help in providing specific advice on soil type, pot size, trimming, and propagation methods.
```

### clarify_or_respond.25 `mirage-#821651`

- Source: `mirage`
- Verifier: `clarify`
- Images: `data/raw/mirage/full/images/MMMT_Decomp/train/MMMT_Decomp-train-002979-01.png`, `data/raw/mirage/full/images/MMMT_Decomp/train/MMMT_Decomp-train-002979-02.png`, `data/raw/mirage/full/images/MMMT_Decomp/train/MMMT_Decomp-train-002979-03.png`

**Prompt**

```text
user: Hello I love on west University parkwy Canterbury area. My balcony faces the back ( I believe its Northwest facing) What can I successfully grow on my balcony and when? Ive included a video of my balcony and picture

Respond using exactly one of these formats:
Decision: clarify
Clarifying question: <one question needed before diagnosis or management>

Decision: respond
Answer: <concise agricultural answer>
```

**Target**

```text
Decision: clarify
Clarifying question: Could you provide more details about how many hours of sunlight your balcony receives each day? Additionally, do you have any preferences for the types of plants you want to grow, such as flowers, vegetables, or herbs? Knowing the size of your balcony and any specific conditions like wind exposure would also be helpful.
```

### clarify_or_respond.26 `mirage-#822664`

- Source: `mirage`
- Verifier: `clarify`
- Images: `data/raw/mirage/full/images/MMMT_Decomp/train/MMMT_Decomp-train-003036-01.png`, `data/raw/mirage/full/images/MMMT_Decomp/train/MMMT_Decomp-train-003036-02.png`, `data/raw/mirage/full/images/MMMT_Decomp/train/MMMT_Decomp-train-003036-03.png`

**Prompt**

```text
user: I was referred to this site by Maryland Farm & Harvest.  I wondered if someone there could help me learn more about some very old poultry waterers we found in a crawl space of a recently purchased house.  They are a little different and smaller than those I found online.  My husband and I would love to know more about them.  I have attached some pictures for your review.  Would someone on your staff be able to assist?  Appreciate your support.  Thank you.

Apologize if this is a duplicate but could not see way to know if my question actually went through in your system.  Did also reach out to your website to confirm receipt but no response there either.
expert: Sorry, I have no experience with old waterers.
user: The only additional information I have is to say that the poultry waterers were found in the Perryville area of Cecil County.  They are approximately 6 inches high and 4 inches wide.   I was going to add this information to the message board but wanted to be sure that's all I needed to do.  Will they see my original message and pictures or do I need to start over and add all that in as well?   Please advise.  Thank you.

Respond using exactly one of these formats:
Decision: clarify
Clarifying question: <one question needed before diagnosis or management>

Decision: respond
Answer: <concise agricultural answer>
```

**Target**

```text
Decision: clarify
Clarifying question: Could you provide more details about the material of the waterers and if there are any identifiable markings or inscriptions on them? Additionally, do you have any information on their estimated age or any known history or previous ownership details?
```

### clarify_or_respond.27 `mirage-#829302`

- Source: `mirage`
- Verifier: `clarify`
- Images: `data/raw/mirage/full/images/MMMT_Decomp/train/MMMT_Decomp-train-001549-01.png`, `data/raw/mirage/full/images/MMMT_Decomp/train/MMMT_Decomp-train-001549-02.png`, `data/raw/mirage/full/images/MMMT_Decomp/train/MMMT_Decomp-train-001549-03.png`

**Prompt**

```text
user: Are you able from attached pictures  identify these small  dry areas on my dwarf Japanese Red Maple tree? This is is 2nd full year of growth. Otherwise seems healthy? located in City of Richmond. Thank you.

Respond using exactly one of these formats:
Decision: clarify
Clarifying question: <one question needed before diagnosis or management>

Decision: respond
Answer: <concise agricultural answer>
```

**Target**

```text
Decision: clarify
Clarifying question: Could you provide more information about the tree's watering schedule, fertilization routine, exposure to sunlight, recent weather conditions, and whether you've noticed any pests or diseases?
```

### clarify_or_respond.28 `mirage-#880875`

- Source: `mirage`
- Verifier: `clarify`
- Images: `data/raw/mirage/full/images/MMMT_Direct/train/MMMT_Direct-train-001447-01.png`, `data/raw/mirage/full/images/MMMT_Direct/train/MMMT_Direct-train-001447-02.png`, `data/raw/mirage/full/images/MMMT_Direct/train/MMMT_Direct-train-001447-03.png`

**Prompt**

```text
user: A number of perennials are displaying signs of distress-stems bending down, newer leaves on tops of plants are smaller, blooms are deformed, newer buds on black-eyed Susans are brown and appear to be dead.  Affected plants include:
goldenrods, coneflowers, cardinal flower, phlox, black-eyed Susans
expert: If there are no other signs of pest or disease damage, the plants are probably suffering from heat stress. Plants are sensitive to both air and soil surface temperatures. Temperatures ranging from 60° to 85°F are optimum for plant growth and development.   Plants have limited ability to regulate temperature and begin to experience stress as air temperatures rise above 86°F.   Water evaporation through pores in the leaves is the primary cooling mechanism used by plants. If a plant doesn’t have enough water, the pores in the leaves close and the plant can no longer keep itself cool. When temperatures are exceptionally high and prolonged, plants can lose water through evaporation much faster than the roots can absorb it.  Wilted, drooping leaves, resulting from a combination of excess water loss and high internal temperatures, are often the first signs of heat stress.   Sunburn is another type of heat damage caused by excessive exposure to direct sunlight. Sunburned leaves and stems become discolored and appear scorched.  Plants experiencing continued heat stress will be unable to maintain the processes necessary for growth and development and will become stunted and vulnerable to insect attack.  So what can be done to minimize effects of heat stress:  Water thoroughly in the morning on days when high temperatures are expected, to help plants cope with excess heat. In general, when watering plants, do it deeply and infrequently so that they put their energy into sending roots down into the lower depths of the soil. During a period of drought, the roots are then able to access moisture reserves deeper in the soil.  If plants are watered too lightly, then the moisture only penetrates the top layer of soil, resulting in plants with shallow root systems.  During a heatwave, the water in the top inch of soil quickly evaporates, so heat-stressed plants also need deep watering.If discovered in time during a heat wave, a drooping wilted plant can often be revived with additional water.   Watch for radiant heat - Pay special attention to plants next to walkways, patios, and other structures that radiate heat. They may be the first to show signs of heat stress and the need for extra water.  Add mulch - Apply two to four inches of organic mulch over the root areas around garden plants, trees, and shrubs. This will lower soil temperature and diminish moisture loss.  Provide shade - Use shade cloth to provide direct sun protection and reduce the air temperature around sun and heat-exposed plants.   Avoid applying fertilizers during extreme heat as they stimulate growth and add to plant stress. Fertilizers also absorb moisture from the root zone.  Many plants will stop blooming during extreme heat. Removing the spent flowers can improve the plant's appearance, keep plants from spending precious resources on fruit and seed development, and set the plant up to bloom again once the heat wave passes.  Following is a link to Iowa State Univ Extension with more info about managing the garden in a heat wave: https://yardandgarden.extension.iastate.edu/how-to/managing-garden-extreme-heat
user: Thanks for your prompt response. I'm sorry I provided you with very skimpy information.  These plants and the damage are confined to one area; other areas of unaffected so far.  We do follow the watering tips and do not fertilize the native pollinators.  Does this point to a fungal disease?  How can I identify whether or not it is a fungal disease?  If it is, what are the treatment options?   Thank you!

Respond using exactly one of these formats:
Decision: clarify
Clarifying question: <one question needed before diagnosis or management>

Decision: respond
Answer: <concise agricultural answer>
```

**Target**

```text
Decision: clarify
Clarifying question: Could you provide more details about the specific environmental conditions in the affected area, such as humidity levels, shade, or proximity to water sources? Additionally, have you noticed any visible fungal growth or spores on the plants?
```

### clarify_or_respond.29 `mirage-#825572`

- Source: `mirage`
- Verifier: `clarify`
- Images: `data/raw/mirage/full/images/MMMT_Decomp/train/MMMT_Decomp-train-001531-01.png`, `data/raw/mirage/full/images/MMMT_Decomp/train/MMMT_Decomp-train-001531-02.png`, `data/raw/mirage/full/images/MMMT_Decomp/train/MMMT_Decomp-train-001531-03.png`

**Prompt**

```text
user: I have this bug that started out brownish now they are coming up green. You smash them and they stink but not stink bug

Respond using exactly one of these formats:
Decision: clarify
Clarifying question: <one question needed before diagnosis or management>

Decision: respond
Answer: <concise agricultural answer>
```

**Target**

```text
Decision: clarify
Clarifying question: Could you provide more details about the size of the bug, the type of plants they are affecting, and any other distinguishing features? Also, knowing the time of year or season might help in identifying the bug.
```

### clarify_or_respond.30 `mirage-#835896`

- Source: `mirage`
- Verifier: `clarify`
- Images: `data/raw/mirage/full/images/MMMT_Direct/train/MMMT_Direct-train-001790-01.png`, `data/raw/mirage/full/images/MMMT_Direct/train/MMMT_Direct-train-001790-02.png`

**Prompt**

```text
user: Hi, my milkweed is in its third year I think, it had self sown elsewhere in yard and I transplanted it here last year. Do you know if this is a disease where I need to pull the whole plant up and throw it out, or is it something more mild? You can see the leaves are very pale and yellowish and most of the orange buds aren’t blooming. Thanks!

Respond using exactly one of these formats:
Decision: clarify
Clarifying question: <one question needed before diagnosis or management>

Decision: respond
Answer: <concise agricultural answer>
```

**Target**

```text
Decision: clarify
Clarifying question: Could you provide more details about any recent changes in the care or environment of your milkweed, such as watering frequency, soil conditions, or any visible pests? Additionally, are there any specific details in the attached images that you would like to highlight?
```

### clarify_or_respond.31 `mirage-#885878`

- Source: `mirage`
- Verifier: `clarify`
- Images: `data/raw/mirage/full/images/MMMT_Direct/train/MMMT_Direct-train-002673-01.png`, `data/raw/mirage/full/images/MMMT_Direct/train/MMMT_Direct-train-002673-02.png`, `data/raw/mirage/full/images/MMMT_Direct/train/MMMT_Direct-train-002673-03.png`

**Prompt**

```text
user: On my property in Blodgett, the Big Leaf Maples are suffering.  The leaves are wilted and more grey than green.  These are trees 60-80’ tall, probably 75 years old or more..  Please, can you advise what the problem is and how it might be treated?  Thank you.

Respond using exactly one of these formats:
Decision: clarify
Clarifying question: <one question needed before diagnosis or management>

Decision: respond
Answer: <concise agricultural answer>
```

**Target**

```text
Decision: clarify
Clarifying question: Could you provide more information about recent weather conditions, watering practices, soil conditions, and any signs of pests or diseases? Additionally, have there been any recent changes in the environment around the trees?
```

### clarify_or_respond.32 `mirage-#876282`

- Source: `mirage`
- Verifier: `clarify`
- Images: `data/raw/mirage/full/images/MMMT_Direct/train/MMMT_Direct-train-001353-01.png`

**Prompt**

```text
user: blueberry bushes about 20 years old look the best they have - so many berries. however when trying to pick the blue ones knocking the plant causes clusters of green berries to drop. is there a way to stop this? I read one source said something about a calcium deficiency - though that is the only info i could find on the web. thank you

Respond using exactly one of these formats:
Decision: clarify
Clarifying question: <one question needed before diagnosis or management>

Decision: respond
Answer: <concise agricultural answer>
```

**Target**

```text
Decision: clarify
Clarifying question: Could you provide more information about your soil conditions, such as recent soil test results, and your current fertilization and watering practices? Additionally, have you noticed any signs of pests or diseases on the plants?
```


## classification

### classification.1 `ip102-ip102-v1-1-images-66967-jpg`

- Source: `ip102`
- Verifier: `label`
- Images: `data/raw/ip102/full/ip102_v1.1/images/66967.jpg`

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
Answer: Lawana imitata Melichar
Evidence: Visible agricultural symptoms or pest features support this label.
```

### classification.2 `ip102-ip102-v1-1-images-67584-jpg`

- Source: `ip102`
- Verifier: `label`
- Images: `data/raw/ip102/full/ip102_v1.1/images/67584.jpg`

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
Answer: Salurnis marginella Guerr
Evidence: Visible agricultural symptoms or pest features support this label.
```

### classification.3 `plantvillage-train-009473`

- Source: `plantvillage`
- Verifier: `label`
- Images: `data/raw/plantvillage/full/images/train/009473.png`

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

### classification.4 `plantvillage-train-000703`

- Source: `plantvillage`
- Verifier: `label`
- Images: `data/raw/plantvillage/full/images/train/000703.png`

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
Answer: apple black rot
Evidence: Visible Apple symptoms support the Black rot label.
```

### classification.5 `plantvillage-train-037285`

- Source: `plantvillage`
- Verifier: `label`
- Images: `data/raw/plantvillage/full/images/train/037285.png`

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
Answer: tomato target spot
Evidence: Visible Tomato symptoms support the Target Spot label.
```

### classification.6 `plantvillage-train-007210`

- Source: `plantvillage`
- Verifier: `label`
- Images: `data/raw/plantvillage/full/images/train/007210.png`

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
Answer: corn maize northern leaf blight
Evidence: Visible Corn (maize) symptoms support the Northern Leaf Blight label.
```

### classification.7 `ip102-ip102-v1-1-images-14410-jpg`

- Source: `ip102`
- Verifier: `label`
- Images: `data/raw/ip102/full/ip102_v1.1/images/14410.jpg`

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
Answer: corn borer
Evidence: Visible agricultural symptoms or pest features support this label.
```

### classification.8 `ip102-ip102-v1-1-images-51849-jpg`

- Source: `ip102`
- Verifier: `label`
- Images: `data/raw/ip102/full/ip102_v1.1/images/51849.jpg`

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
Answer: Cicadella viridis
Evidence: Visible agricultural symptoms or pest features support this label.
```

### classification.9 `plantvillage-train-037989`

- Source: `plantvillage`
- Verifier: `label`
- Images: `data/raw/plantvillage/full/images/train/037989.png`

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
Answer: tomato tomato yellow leaf curl virus
Evidence: Visible Tomato symptoms support the Tomato Yellow Leaf Curl Virus label.
```

### classification.10 `rice_disease-train-013583`

- Source: `rice_disease`
- Verifier: `label`
- Images: `data/raw/rice_disease/full/images/train/013583.jpg`

**Prompt**

```text
Identify the crop issue or pest in this agricultural image.

Choose exactly one label from this allowed label set:
Allowed labels: bacterial leaf blight; brown plant hopper; brown spot; chilo suppressalis; false smut; healthy; hispa; leaf scald; narrow brown spot; nitrogen deficiency; phosphorus deficiency; potassium deficiency; rice bakanae disease; rice gall midge; rice leaf and neck blast; rice leaf folder; rice skipper; rice yellow stem borer; sheath blight; thrips; tungro virus
Respond in this format:
Answer: <one allowed label>
Evidence: <brief visible symptom evidence>
Do not leave Answer blank, invent labels, or copy the placeholder text.
```

**Target**

```text
Answer: tungro virus
Evidence: Visible Rice symptoms support the Tungro virus label.
```

### classification.11 `ip102-ip102-v1-1-images-60546-jpg`

- Source: `ip102`
- Verifier: `label`
- Images: `data/raw/ip102/full/ip102_v1.1/images/60546.jpg`

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
Answer: Ceroplastes rubens
Evidence: Visible agricultural symptoms or pest features support this label.
```

### classification.12 `plantvillage-train-042620`

- Source: `plantvillage`
- Verifier: `label`
- Images: `data/raw/plantvillage/full/images/train/042620.png`

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

### classification.13 `plantvillage-train-000188`

- Source: `plantvillage`
- Verifier: `label`
- Images: `data/raw/plantvillage/full/images/train/000188.png`

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
Answer: apple apple scab
Evidence: Visible Apple symptoms support the Apple scab label.
```

### classification.14 `plantvillage-train-028608`

- Source: `plantvillage`
- Verifier: `label`
- Images: `data/raw/plantvillage/full/images/train/028608.png`

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
Answer: strawberry healthy
Evidence: Visible Strawberry symptoms support the healthy label.
```

### classification.15 `plantdoc-142`

- Source: `plantdoc`
- Verifier: `label`
- Images: `data/raw/plantdoc/full/images/train/000043.png`

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
Answer: tomato mold leaf
Evidence: Visible agricultural symptoms or pest features support this label.
```

### classification.16 `plantvillage-train-001649`

- Source: `plantvillage`
- Verifier: `label`
- Images: `data/raw/plantvillage/full/images/train/001649.png`

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
Answer: apple healthy
Evidence: Visible Apple symptoms support the healthy label.
```

### classification.17 `plantvillage-train-000100`

- Source: `plantvillage`
- Verifier: `label`
- Images: `data/raw/plantvillage/full/images/train/000100.png`

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
Answer: apple apple scab
Evidence: Visible Apple symptoms support the Apple scab label.
```

### classification.18 `plantvillage-train-025257`

- Source: `plantvillage`
- Verifier: `label`
- Images: `data/raw/plantvillage/full/images/train/025257.png`

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
Answer: soybean healthy
Evidence: Visible Soybean symptoms support the healthy label.
```

### classification.19 `ip102-ip102-v1-1-images-11416-jpg`

- Source: `ip102`
- Verifier: `label`
- Images: `data/raw/ip102/full/ip102_v1.1/images/11416.jpg`

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
Answer: wireworm
Evidence: Visible agricultural symptoms or pest features support this label.
```

### classification.20 `plantvillage-train-021825`

- Source: `plantvillage`
- Verifier: `label`
- Images: `data/raw/plantvillage/full/images/train/021825.png`

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
Answer: potato healthy
Evidence: Visible Potato symptoms support the healthy label.
```

### classification.21 `rice_disease-train-026398`

- Source: `rice_disease`
- Verifier: `label`
- Images: `data/raw/rice_disease/full/images/train/026398.jpg`

**Prompt**

```text
Identify the crop issue or pest in this agricultural image.

Choose exactly one label from this allowed label set:
Allowed labels: bacterial leaf blight; brown plant hopper; brown spot; chilo suppressalis; false smut; healthy; hispa; leaf scald; narrow brown spot; nitrogen deficiency; phosphorus deficiency; potassium deficiency; rice bakanae disease; rice gall midge; rice leaf and neck blast; rice leaf folder; rice skipper; rice yellow stem borer; sheath blight; thrips; tungro virus
Respond in this format:
Answer: <one allowed label>
Evidence: <brief visible symptom evidence>
Do not leave Answer blank, invent labels, or copy the placeholder text.
```

**Target**

```text
Answer: bacterial leaf blight
Evidence: Visible Rice symptoms support the Bacterial leaf blight label.
```

### classification.22 `plantdoc-6234`

- Source: `plantdoc`
- Verifier: `label`
- Images: `data/raw/plantdoc/full/images/train/001725.png`

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
Answer: corn gray leaf spot
Evidence: Visible agricultural symptoms or pest features support this label.
```

### classification.23 `ip102-ip102-v1-1-images-01802-jpg`

- Source: `ip102`
- Verifier: `label`
- Images: `data/raw/ip102/full/ip102_v1.1/images/01802.jpg`

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

### classification.24 `plantvillage-train-005660`

- Source: `plantvillage`
- Verifier: `label`
- Images: `data/raw/plantvillage/full/images/train/005660.png`

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
Answer: corn maize cercospora leaf spot gray leaf spot
Evidence: Visible Corn (maize) symptoms support the Cercospora leaf spot Gray leaf spot label.
```

### classification.25 `ip102-ip102-v1-1-images-24056-jpg`

- Source: `ip102`
- Verifier: `label`
- Images: `data/raw/ip102/full/ip102_v1.1/images/24056.jpg`

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
Answer: bird cherry oataphid
Evidence: Visible agricultural symptoms or pest features support this label.
```

### classification.26 `rice_disease-train-026362`

- Source: `rice_disease`
- Verifier: `label`
- Images: `data/raw/rice_disease/full/images/train/026362.jpg`

**Prompt**

```text
Identify the crop issue or pest in this agricultural image.

Choose exactly one label from this allowed label set:
Allowed labels: bacterial leaf blight; brown plant hopper; brown spot; chilo suppressalis; false smut; healthy; hispa; leaf scald; narrow brown spot; nitrogen deficiency; phosphorus deficiency; potassium deficiency; rice bakanae disease; rice gall midge; rice leaf and neck blast; rice leaf folder; rice skipper; rice yellow stem borer; sheath blight; thrips; tungro virus
Respond in this format:
Answer: <one allowed label>
Evidence: <brief visible symptom evidence>
Do not leave Answer blank, invent labels, or copy the placeholder text.
```

**Target**

```text
Answer: rice gall midge
Evidence: Visible Rice symptoms support the Rice Gall Midge label.
```

### classification.27 `plantvillage-train-004457`

- Source: `plantvillage`
- Verifier: `label`
- Images: `data/raw/plantvillage/full/images/train/004457.png`

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
Answer: cherry including sour powdery mildew
Evidence: Visible Cherry (including sour) symptoms support the Powdery mildew label.
```

### classification.28 `plantdoc-4356`

- Source: `plantdoc`
- Verifier: `label`
- Images: `data/raw/plantdoc/full/images/train/001211.png`

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
Answer: peach leaf
Evidence: Visible agricultural symptoms or pest features support this label.
```

### classification.29 `ip102-ip102-v1-1-images-08778-jpg`

- Source: `ip102`
- Verifier: `label`
- Images: `data/raw/ip102/full/ip102_v1.1/images/08778.jpg`

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
Answer: grub
Evidence: Visible agricultural symptoms or pest features support this label.
```

### classification.30 `plantvillage-train-029418`

- Source: `plantvillage`
- Verifier: `label`
- Images: `data/raw/plantvillage/full/images/train/029418.png`

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

### classification.31 `plantvillage-train-027723`

- Source: `plantvillage`
- Verifier: `label`
- Images: `data/raw/plantvillage/full/images/train/027723.png`

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
Answer: strawberry leaf scorch
Evidence: Visible Strawberry symptoms support the Leaf scorch label.
```

### classification.32 `plantvillage-train-019768`

- Source: `plantvillage`
- Verifier: `label`
- Images: `data/raw/plantvillage/full/images/train/019768.png`

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
Answer: pepper bell healthy
Evidence: Visible Pepper, bell symptoms support the healthy label.
```


## consultation

### consultation.1 `agbase-agbase-737020`

- Source: `agbase`
- Verifier: `structured`
- Images: `data/raw/agbase/full/737020/737020_1.jpg`, `data/raw/agbase/full/737020/737020_2.jpg`, `data/raw/agbase/full/737020/737020_3.jpg`

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
Diagnosis: human
Evidence: Visible symptoms should be checked against the image and crop context.
Uncertainty: Image-only assessment; confirm with field context before treatment.
Management: capture or take a picture of the critter for identification, consider consulting a good exterminator
Follow-up: Share close-up images and crop history if symptoms progress or the diagnosis is uncertain.
```

### consultation.2 `agbase-agbase-851455`

- Source: `agbase`
- Verifier: `structured`
- Images: `data/raw/agbase/full/851455/851455_1.jpg`, `data/raw/agbase/full/851455/851455_2.jpg`, `data/raw/agbase/full/851455/851455_3.jpg`

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
Diagnosis: apple, pear, cherry, plum
Evidence: Visible symptoms should be checked against the image and crop context.
Uncertainty: Image-only assessment; confirm with field context before treatment.
Management: No source-specific management step was provided; recommend local extension follow-up.
Follow-up: Share close-up images and crop history if symptoms progress or the diagnosis is uncertain.
```

### consultation.3 `mirage-#399473`

- Source: `mirage`
- Verifier: `exact_match`
- Images: `data/raw/mirage/full/images/MMST_Standard/train/MMST_Standard-train-010451-01.png`, `data/raw/mirage/full/images/MMST_Standard/train/MMST_Standard-train-010451-02.png`

**Prompt**

```text
Please answer the following user question. You should first analyze the provided image, mentioning any visible clues or observations. Then, present the identification result. Write the entire content as one coherent paragraph (analysis + results).
User: Please provide plant IDs for the plants shown in the images taken at Cylburn Arboretum.

Respond using these line-start section headers exactly once:
Diagnosis:
Evidence:
Uncertainty:
Management:
Follow-up:
```

**Target**

```text
Diagnosis: The first image shows a climbing vine with clusters of tubular, bright pinkish-red flowers that flare slightly at the tips and have yellow stamens protruding, along with dark green, ovate leaves with smooth edges. This floral shape and vine habit clearly correspond to an ornamental honeysuckle, specifically Lonicera x brownii 'Dropmore Scarlet', known for its striking, long-lasting summer blooms that attract pollinators like hummingbirds. The second image features a close-up of a single, light pink flower with darker pink veins radiating prominently from the center, surrounded by deeply lobed, palmate green leaves. These characteristics are typical of the perennial geranium called 'Striped Bloody' (Geranium sanguineum var. striatum), a low-growing garden favorite valued for its dainty veined flowers and finely divided foliage, often used as groundcover. Together, these plants—an ornamental honeysuckle vine and a striped perennial geranium—are well-suited to garden settings like Cylburn Arboretum, combining vibrant color and texture.
Evidence: Visible symptoms should be checked against the image and crop context.
Uncertainty: Image-only assessment; confirm with field context before treatment.
Management: No source-specific management step was provided; recommend local extension follow-up.
Follow-up: Share close-up images and crop history if symptoms progress or the diagnosis is uncertain.
```

### consultation.4 `agbase-agbase-641133`

- Source: `agbase`
- Verifier: `structured`
- Images: `data/raw/agbase/full/641133/641133_1.jpg`, `data/raw/agbase/full/641133/641133_2.jpg`

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
Diagnosis: mold
Evidence: white patches on soil
Uncertainty: Image-only assessment; confirm with field context before treatment.
Management: remove from the soil and discard it. keep the mulch away from the base of the stems.
Follow-up: Share close-up images and crop history if symptoms progress or the diagnosis is uncertain.
```

### consultation.5 `agbase-agbase-840431`

- Source: `agbase`
- Verifier: `structured`
- Images: `data/raw/agbase/full/840431/840431_2.jpg`, `data/raw/agbase/full/840431/840431_3.jpg`, `data/raw/agbase/full/840431/840431_1.jpg`

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
Diagnosis: plant (general, possibly trefoil), beef cattle
Evidence: Visible symptoms should be checked against the image and crop context.
Uncertainty: Image-only assessment; confirm with field context before treatment.
Management: No source-specific management step was provided; recommend local extension follow-up.
Follow-up: Share close-up images and crop history if symptoms progress or the diagnosis is uncertain.
```

### consultation.6 `agbase-agbase-621967`

- Source: `agbase`
- Verifier: `structured`
- Images: `data/raw/agbase/full/621967/621967_3.jpg`, `data/raw/agbase/full/621967/621967_2.jpg`, `data/raw/agbase/full/621967/621967_1.jpg`

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
Diagnosis: bear's breeches (acanthus mollis)
Evidence: Visible symptoms should be checked against the image and crop context.
Uncertainty: Image-only assessment; confirm with field context before treatment.
Management: No source-specific management step was provided; recommend local extension follow-up.
Follow-up: Share close-up images and crop history if symptoms progress or the diagnosis is uncertain.
```

### consultation.7 `agbase-agbase-793151`

- Source: `agbase`
- Verifier: `structured`
- Images: `data/raw/agbase/full/793151/793151_2.jpg`, `data/raw/agbase/full/793151/793151_1.jpg`

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
Diagnosis: creeping field cress
Evidence: Visible symptoms should be checked against the image and crop context.
Uncertainty: Image-only assessment; confirm with field context before treatment.
Management: deprive the weed of its ability to photosynthesize by whacking all the leaves as soon as they emerge in the spring. then, just before summer perennials go dormant, chop it back again. another option is to spray the entire area with glyphosate, a broad-spectrum weed killer, following the label instructions and taking safety precautions. alternatively, relocate your vegetable garden to an area that is not infested with the fieldcress.
Follow-up: Share close-up images and crop history if symptoms progress or the diagnosis is uncertain.
```

### consultation.8 `mirage-#841773`

- Source: `mirage`
- Verifier: `exact_match`
- Images: `data/raw/mirage/full/images/MMST_Standard/train/MMST_Standard-train-010557-01.png`

**Prompt**

```text
Please answer the following user question. You should first analyze the provided image, mentioning any visible clues or observations. Then, present the identification result. Write the entire content as one coherent paragraph (analysis + results).
User: Plant ID: Is this poisonous pokeweed? It is about 5 feet tall, growing in next door neighbor's yard in Battle Creek.

Respond using these line-start section headers exactly once:
Diagnosis:
Evidence:
Uncertainty:
Management:
Follow-up:
```

**Target**

```text
Diagnosis: The image clearly shows a tall, robust plant approximately 5 feet in height with broad, ovate green leaves and numerous elongated clusters of small white to pinkish flowers hanging downward. The stems appear sturdy and somewhat reddish in color. These characteristics match those of common pokeweed (Phytolacca americana L.), a native North American perennial known for its distinctive drooping flower clusters and large leaves. Common pokeweed is indeed toxic, with all parts causing potential stomach upset if ingested, particularly the berries, roots, and mature stems and leaves. Its presence in your neighbor’s yard in Battle Creek aligns well with its typical habitat and growth pattern. Because of its toxicity and persistence, care should be taken around this plant, especially to prevent accidental ingestion by children or pets.
Evidence: Visible symptoms should be checked against the image and crop context.
Uncertainty: Image-only assessment; confirm with field context before treatment.
Management: No source-specific management step was provided; recommend local extension follow-up.
Follow-up: Share close-up images and crop history if symptoms progress or the diagnosis is uncertain.
```

### consultation.9 `agbase-agbase-869265`

- Source: `agbase`
- Verifier: `structured`
- Images: `data/raw/agbase/full/869265/869265_1.jpg`

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
Diagnosis: sunscald and frost cracks
Evidence: losing bark, buckling lower bark
Uncertainty: Image-only assessment; confirm with field context before treatment.
Management: pull back the mulch about 6-8” from the tree
Follow-up: Share close-up images and crop history if symptoms progress or the diagnosis is uncertain.
```

### consultation.10 `mirage-#869048`

- Source: `mirage`
- Verifier: `exact_match`
- Images: `data/raw/mirage/full/images/MMST_Standard/train/MMST_Standard-train-013423-01.png`

**Prompt**

```text
Is this plant a weed, and what should I do with it? I moved into a new house, and the backyard along the fence line is covered with these, and I have not been able to determine what they are. They grow straight up in clusters, and have a large root ball with thin, hair-like roots.

Respond using these line-start section headers exactly once:
Diagnosis:
Evidence:
Uncertainty:
Management:
Follow-up:
```

**Target**

```text
Diagnosis: The plant in your photo and description is Motherwort (Leonurus cardiaca), a perennial weed from the mint family that often invades yards, gardens, and disturbed soils. It can form dense colonies along fences and is known for rapidly spreading via rhizomes (underground stems), which explains the large root balls and clusters you're seeing. The leaves of Motherwort are deeply lobed, often resembling maple leaves, and the stems are characteristically square and hairy. It typically grows 2 to 4 feet tall.

Motherwort is weedy and invasive, crowding out native and garden plants and making it difficult for them to thrive. It produces spiny calyces that persist into winter and can be sharp and uncomfortable to handle. Many gardeners and land managers report it is challenging to control, as it returns year after year if not fully removed.

Management:
- You should remove Motherwort from your yard to prevent it from spreading further.
- Wear long sleeves and gloves when handling the plant, as its spiny parts can cause a rash or discomfort if they come into contact with your skin.
- Pull out the entire root system, including the root ball and thin, hair-like roots, as incomplete removal can allow it to regrow.

While bees are attracted to its flowers, there are many native plants more suitable for supporting pollinators without the risk of invasive spread. After removal, consider replacing Motherwort with pollinator-friendly native species. Consistent hand-pulling over several seasons may be needed for full eradication.
Evidence: Visible symptoms should be checked against the image and crop context.
Uncertainty: Image-only assessment; confirm with field context before treatment.
Management: No source-specific management step was provided; recommend local extension follow-up.
Follow-up: Share close-up images and crop history if symptoms progress or the diagnosis is uncertain.
```

### consultation.11 `agbase-agbase-833497`

- Source: `agbase`
- Verifier: `structured`
- Images: `data/raw/agbase/full/833497/833497_1.jpg`, `data/raw/agbase/full/833497/833497_2.jpg`, `data/raw/agbase/full/833497/833497_3.jpg`

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
Diagnosis: black knot
Evidence: Visible symptoms should be checked against the image and crop context.
Uncertainty: Image-only assessment; confirm with field context before treatment.
Management: No source-specific management step was provided; recommend local extension follow-up.
Follow-up: Share close-up images and crop history if symptoms progress or the diagnosis is uncertain.
```

### consultation.12 `agbase-agbase-742869`

- Source: `agbase`
- Verifier: `structured`
- Images: `data/raw/agbase/full/742869/742869_3.jpg`, `data/raw/agbase/full/742869/742869_2.jpg`, `data/raw/agbase/full/742869/742869_1.jpg`

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
Diagnosis: cottonwood, norway maple
Evidence: leaves did not drop in fall
Uncertainty: Image-only assessment; confirm with field context before treatment.
Management: check for bud formation, water according to guidelines in attached fact sheet
Follow-up: Share close-up images and crop history if symptoms progress or the diagnosis is uncertain.
```

### consultation.13 `agrillava-agrillava-001713`

- Source: `agrillava`
- Verifier: `structured`
- Images: `data/raw/agrillava/full/Img/wheat_longlegged_spider_mite_99.jpg`

**Prompt**

```text
Which part of the insect in the picture is orange yellow?

Respond using these line-start section headers exactly once:
Diagnosis:
Evidence:
Uncertainty:
Management:
Follow-up:
```

**Target**

```text
Diagnosis: Legs
Evidence: Visible symptoms should be checked against the image and crop context.
Uncertainty: Image-only assessment; confirm with field context before treatment.
Management: No source-specific management step was provided; recommend local extension follow-up.
Follow-up: Share close-up images and crop history if symptoms progress or the diagnosis is uncertain.
```

### consultation.14 `agbase-agbase-853348`

- Source: `agbase`
- Verifier: `structured`
- Images: `data/raw/agbase/full/853348/853348_1.jpg`, `data/raw/agbase/full/853348/853348_2.jpg`

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
Diagnosis: douglas fir
Evidence: dried branches and dead trees
Uncertainty: Image-only assessment; confirm with field context before treatment.
Management: pull the dead tree up and examine the roots, cut into the lower stem with a knife and look for distinctive black streaking/staining of the wood, keep competing vegetation controlled and hope for normal rain
Follow-up: Share close-up images and crop history if symptoms progress or the diagnosis is uncertain.
```

### consultation.15 `mirage-#848788`

- Source: `mirage`
- Verifier: `exact_match`
- Images: `data/raw/mirage/full/images/MMST_Standard/train/MMST_Standard-train-011957-01.png`

**Prompt**

```text
Please answer the following user question. You should first analyze the provided image, mentioning any visible clues or observations. Then, present the identification result. Write the entire content as one coherent paragraph (analysis + results).
User: My volunteer cosmos are growing super weird this year. Basically, they aren't growing petals but their seed areas are elongated and bizarre. I don't know how to describe it to do a search on it. I'm worried I have some sort of disease going on here. Any insight?

Respond using these line-start section headers exactly once:
Diagnosis:
Evidence:
Uncertainty:
Management:
Follow-up:
```

**Target**

```text
Diagnosis: The photo shows cosmos flowers with abnormal, distorted blooms that lack the typical colorful petals, instead presenting elongated, green, brushy structures around the central seed areas. This unusual floral development is a clear symptom of aster yellows, a disease caused by a phytoplasma that infects the plant's phloem. The presence of these malformed flower heads, along with the lack of normal petal formation, matches the classic signs of this systemic disease transmitted by the aster leafhopper insect. Unfortunately, aster yellows cannot be cured once plants are infected; the best course of action is to promptly remove and destroy these affected cosmos to prevent further spread. Additionally, addressing weed hosts nearby that may harbor the leafhopper could help limit transmission. For future plantings, monitoring for early symptoms and removing infected plants remains the most effective management strategy.
Evidence: Visible symptoms should be checked against the image and crop context.
Uncertainty: Image-only assessment; confirm with field context before treatment.
Management: No source-specific management step was provided; recommend local extension follow-up.
Follow-up: Share close-up images and crop history if symptoms progress or the diagnosis is uncertain.
```

### consultation.16 `agbase-agbase-598380`

- Source: `agbase`
- Verifier: `structured`
- Images: `data/raw/agbase/full/598380/598380_2.jpg`, `data/raw/agbase/full/598380/598380_1.jpg`, `data/raw/agbase/full/598380/598380_3.jpg`

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
Diagnosis: ninebark
Evidence: leaves suddenly falling off
Uncertainty: Image-only assessment; confirm with field context before treatment.
Management: check the soil moisture and water deeply, do not overwater, probe with a screwdriver and water when the top several inches of the soil dries out slightly, repot into a larger container in the spring or plant in the ground if space is available
Follow-up: Share close-up images and crop history if symptoms progress or the diagnosis is uncertain.
```

### consultation.17 `agbase-agbase-828964`

- Source: `agbase`
- Verifier: `structured`
- Images: `data/raw/agbase/full/828964/828964_1.jpg`, `data/raw/agbase/full/828964/828964_2.jpg`

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
Evidence: powdery mildew
Uncertainty: Image-only assessment; confirm with field context before treatment.
Management: prune out and dispose of all the mildew afflicted stems with or without flowers and damaged leaves, increase air circulation between the plants with spacing, apply preventative contact fungicide, consider shearing the whole plant down to regrow fresh leaves
Follow-up: Share close-up images and crop history if symptoms progress or the diagnosis is uncertain.
```

### consultation.18 `agbase-agbase-764153`

- Source: `agbase`
- Verifier: `structured`
- Images: `data/raw/agbase/full/764153/764153_2.jpg`, `data/raw/agbase/full/764153/764153_3.jpg`, `data/raw/agbase/full/764153/764153_1.jpg`

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
Diagnosis: chemical burn
Evidence: straight lines of damage, areas that are less intensely affected and hot spots where the compound accumulated
Uncertainty: Image-only assessment; confirm with field context before treatment.
Management: water the lawn about 1" per week
Follow-up: Share close-up images and crop history if symptoms progress or the diagnosis is uncertain.
```

### consultation.19 `agrillava-agrillava-000409`

- Source: `agrillava`
- Verifier: `structured`
- Images: `data/raw/agrillava/full/Img/citrus_bactrocera_tsuneonis_31.jpg`

**Prompt**

```text
What types of fruits are mainly harmed by Bactrocera tsuneonis?

Respond using these line-start section headers exactly once:
Diagnosis:
Evidence:
Uncertainty:
Management:
Follow-up:
```

**Target**

```text
Diagnosis: Citrus species
Evidence: Visible symptoms should be checked against the image and crop context.
Uncertainty: Image-only assessment; confirm with field context before treatment.
Management: No source-specific management step was provided; recommend local extension follow-up.
Follow-up: Share close-up images and crop history if symptoms progress or the diagnosis is uncertain.
```

### consultation.20 `agbase-agbase-871508`

- Source: `agbase`
- Verifier: `structured`
- Images: `data/raw/agbase/full/871508/871508_3.jpg`, `data/raw/agbase/full/871508/871508_1.jpg`

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
Diagnosis: leaftier moth on hydrangea. four-lined plant bug damage on phlox. possible aphid damage and black spot on rose
Evidence: leaves turn a darker color and are deformed, get rippled and bent, get spots then the leaf dies. chew holes on some plants. stunted growth and puckered leaves on rose
Uncertainty: Image-only assessment; confirm with field context before treatment.
Management: handpick leaftier moth structures or mechanically squish the pest, check for four-lined plant bugs in late may and early june, remove damaged plants in the fall, consider spraying insecticidal soap or pyrethrins, use sanitation and good air circulation to manage black spot on rose, restrict irrigation during cloudy, humid weather, use labeled fungicide sprays every 7-14 days
Follow-up: Share close-up images and crop history if symptoms progress or the diagnosis is uncertain.
```

### consultation.21 `agbase-agbase-648056`

- Source: `agbase`
- Verifier: `structured`
- Images: `data/raw/agbase/full/648056/648056_2.jpg`, `data/raw/agbase/full/648056/648056_1.jpg`, `data/raw/agbase/full/648056/648056_3.jpg`

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
Diagnosis: petunia
Evidence: Visible symptoms should be checked against the image and crop context.
Uncertainty: Image-only assessment; confirm with field context before treatment.
Management: No source-specific management step was provided; recommend local extension follow-up.
Follow-up: Share close-up images and crop history if symptoms progress or the diagnosis is uncertain.
```

### consultation.22 `mirage-#792847`

- Source: `mirage`
- Verifier: `exact_match`
- Images: `data/raw/mirage/full/images/MMST_Standard/train/MMST_Standard-train-014642-01.png`, `data/raw/mirage/full/images/MMST_Standard/train/MMST_Standard-train-014642-02.png`, `data/raw/mirage/full/images/MMST_Standard/train/MMST_Standard-train-014642-03.png`

**Prompt**

```text
I have a spider plant hanging in a south-facing window with sheer curtains. I've noticed the plant looks light in color. Is it getting too much sun? Should I no longer hang it in the window?

Respond using these line-start section headers exactly once:
Diagnosis:
Evidence:
Uncertainty:
Management:
Follow-up:
```

**Target**

```text
Diagnosis: Your spider plant appears healthy overall, though the light green color you observe is common for this species. Spider plants (Chlorophytum comosum) naturally grow in indirect sunlight as understory plants. While they thrive in bright, indirect light, exposure to direct sunlight can result in leaf scorch or lighter leaf color. Since your plant is hanging in a south-facing window with sheer curtains, it is likely receiving filtered light, which is typically suitable. However, if the leaves continue to look lighter or begin to develop brown tips or scorched spots, this could indicate too much light.

To ensure optimal health, maintain the plant in bright, indirect sunlight and avoid direct exposure. Other care tips include using a pot with drainage holes, letting the soil dry out slightly before watering, using distilled water if possible, fertilizing monthly at half strength during periods of active growth, and repotting when the plant becomes crowded. Monitor your plant for signs of stress, but with your current setup and regular care, your spider plant should continue to thrive.
Evidence: Visible symptoms should be checked against the image and crop context.
Uncertainty: Image-only assessment; confirm with field context before treatment.
Management: No source-specific management step was provided; recommend local extension follow-up.
Follow-up: Share close-up images and crop history if symptoms progress or the diagnosis is uncertain.
```

### consultation.23 `mirage-#719759`

- Source: `mirage`
- Verifier: `exact_match`
- Images: `data/raw/mirage/full/images/MMST_Standard/train/MMST_Standard-train-012824-01.png`, `data/raw/mirage/full/images/MMST_Standard/train/MMST_Standard-train-012824-02.png`, `data/raw/mirage/full/images/MMST_Standard/train/MMST_Standard-train-012824-03.png`

**Prompt**

```text
What is the powdery looking stuff on my zucchini and yellow squash plants that eventually kills the leaves? Is this a squash plant disease? What is the solution?

Respond using these line-start section headers exactly once:
Diagnosis:
Evidence:
Uncertainty:
Management:
Follow-up:
```

**Target**

```text
Diagnosis: The powdery looking substance on your zucchini and yellow squash plants is powdery mildew, a common and easily recognized fungal disease. It appears as spots or patches of white to grayish, talcum-powder-like growth, mostly on the upper sides of leaves but can also affect the bottom sides, young stems, buds, flowers, and young fruit. Infected leaves may become distorted, turn yellow with patches of green, and eventually die and fall off.

Powdery mildew thrives in warm, dry climates and crowded plantings with poor air circulation, particularly in areas with high relative humidity. Young, succulent plant tissue is especially susceptible.

To manage powdery mildew:
- Plant resistant or tolerant varieties if available, and avoid shady, crowded locations.
- Remove and destroy all infected plant parts as soon as they appear to reduce spread and overwintering of the fungus. Do not compost infected debris.
- Prune overcrowded plants to improve air circulation and reduce humidity around the foliage.
- Avoid overhead watering and late-summer applications of nitrogen fertilizer.

If the disease persists:
- Consider applying fungicides such as sulfur, neem oil, potassium bicarbonate, or other labeled products, following all label instructions. Fungicides are most effective when combined with the cultural practices above and should be applied every 7 to 14 days as needed throughout the growing season.

An alternative, experimental control involves spraying a solution of one tablespoon baking soda plus 2.5 tablespoons lightweight horticultural oil in one gallon of water. Use this at your own risk, as its effectiveness may vary.

By consistently following these steps, you can help reduce the impact of powdery mildew and maintain healthier squash plants.
Evidence: Visible symptoms should be checked against the image and crop context.
Uncertainty: Image-only assessment; confirm with field context before treatment.
Management: No source-specific management step was provided; recommend local extension follow-up.
Follow-up: Share close-up images and crop history if symptoms progress or the diagnosis is uncertain.
```

### consultation.24 `mirage-#759041`

- Source: `mirage`
- Verifier: `exact_match`
- Images: `data/raw/mirage/full/images/MMST_Standard/train/MMST_Standard-train-016803-01.png`

**Prompt**

```text
Can you help me interpret my soil test report? I'd especially like to know what can cause the Magnesium and Calcium to be in the excessive range. Is that due to the high pH? Any insights are welcome.

Respond using these line-start section headers exactly once:
Diagnosis:
Evidence:
Uncertainty:
Management:
Follow-up:
```

**Target**

```text
Diagnosis: High calcium and magnesium are normal and not a problem. It is due to the rock (parent material) from which the soil is derived.
Evidence: Visible symptoms should be checked against the image and crop context.
Uncertainty: Image-only assessment; confirm with field context before treatment.
Management: No source-specific management step was provided; recommend local extension follow-up.
Follow-up: Share close-up images and crop history if symptoms progress or the diagnosis is uncertain.
```

### consultation.25 `agbase-agbase-804321`

- Source: `agbase`
- Verifier: `structured`
- Images: `data/raw/agbase/full/804321/804321_1.jpg`

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
Diagnosis: lilac bush
Evidence: Visible symptoms should be checked against the image and crop context.
Uncertainty: Image-only assessment; confirm with field context before treatment.
Management: No source-specific management step was provided; recommend local extension follow-up.
Follow-up: Share close-up images and crop history if symptoms progress or the diagnosis is uncertain.
```

### consultation.26 `agbase-agbase-660790`

- Source: `agbase`
- Verifier: `structured`
- Images: `data/raw/agbase/full/660790/660790_2.jpg`, `data/raw/agbase/full/660790/660790_1.jpg`

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
Diagnosis: bottlebrush buckeye (aesculus parviflora)
Evidence: Visible symptoms should be checked against the image and crop context.
Uncertainty: Image-only assessment; confirm with field context before treatment.
Management: No source-specific management step was provided; recommend local extension follow-up.
Follow-up: Share close-up images and crop history if symptoms progress or the diagnosis is uncertain.
```

### consultation.27 `agbase-agbase-864086`

- Source: `agbase`
- Verifier: `structured`
- Images: `data/raw/agbase/full/864086/864086_1.jpg`

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
Diagnosis: arborvitae, quince
Evidence: Visible symptoms should be checked against the image and crop context.
Uncertainty: Image-only assessment; confirm with field context before treatment.
Management: No source-specific management step was provided; recommend local extension follow-up.
Follow-up: Share close-up images and crop history if symptoms progress or the diagnosis is uncertain.
```

### consultation.28 `agbase-agbase-717457`

- Source: `agbase`
- Verifier: `structured`
- Images: `data/raw/agbase/full/717457/717457_1.jpg`, `data/raw/agbase/full/717457/717457_2.jpg`

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
Diagnosis: caterpillar (funerary dagger acronicta funeralis)
Evidence: Visible symptoms should be checked against the image and crop context.
Uncertainty: Image-only assessment; confirm with field context before treatment.
Management: No source-specific management step was provided; recommend local extension follow-up.
Follow-up: Share close-up images and crop history if symptoms progress or the diagnosis is uncertain.
```

### consultation.29 `mirage-#577613`

- Source: `mirage`
- Verifier: `exact_match`
- Images: `data/raw/mirage/full/images/MMST_Standard/train/MMST_Standard-train-003820-01.png`, `data/raw/mirage/full/images/MMST_Standard/train/MMST_Standard-train-003820-02.png`

**Prompt**

```text
How do I bring my dill plant and pineapple sage plant back to life? They are both wilting?

Respond using these line-start section headers exactly once:
Diagnosis:
Evidence:
Uncertainty:
Management:
Follow-up:
```

**Target**

```text
Diagnosis: Both of these houseplants look too deteriorated to save. We recommend leaving them alone (not watering, since over-watering is the most common way to kill houseplants) and see if they begin to leaf out. Give them as much sun as you can. If they do not leaf out, discard them. They are not rare plants and should not be difficult to replace.
Evidence: Visible symptoms should be checked against the image and crop context.
Uncertainty: Image-only assessment; confirm with field context before treatment.
Management: No source-specific management step was provided; recommend local extension follow-up.
Follow-up: Share close-up images and crop history if symptoms progress or the diagnosis is uncertain.
```

### consultation.30 `mirage-#460192`

- Source: `mirage`
- Verifier: `exact_match`
- Images: `data/raw/mirage/full/images/MMST_Standard/train/MMST_Standard-train-004406-01.png`

**Prompt**

```text
Please answer the following user question. You should first analyze the provided image, mentioning any visible clues or observations. Then, present the identification result. Write the entire content as one coherent paragraph (analysis + results).
User: Can you please tell me what these shrubs in my yard are? One is about 3.5 feet tall and the other is about 1 foot tall. They just showed up one year. They are evergreen with shiny leaves, and the berries are black, clustered, and oblong-ish.

Respond using these line-start section headers exactly once:
Diagnosis:
Evidence:
Uncertainty:
Management:
Follow-up:
```

**Target**

```text
Diagnosis: The image shows a dense shrub with glossy, dark green evergreen leaves that are smooth-edged and arranged alternately. Among the foliage, there are tight clusters of small, shiny, black oblong berries hanging from the stems, matching your description of the plants in your yard. These visual clues, combined with the size and appearance you provided, identify the shrubs as Spurge Laurel (Daphne laureola L.). This species is a common evergreen landscape shrub that can spread invasively. It is important to note that both the sap and seeds of Spurge Laurel are toxic, so handle the plants with care, especially if you plan to remove them. Pulling out young plants is often effective, and discarded material should be placed in the trash to prevent re-sprouting, as composting or yard waste bins may lead to further spread.
Evidence: Visible symptoms should be checked against the image and crop context.
Uncertainty: Image-only assessment; confirm with field context before treatment.
Management: No source-specific management step was provided; recommend local extension follow-up.
Follow-up: Share close-up images and crop history if symptoms progress or the diagnosis is uncertain.
```

### consultation.31 `agbase-agbase-831976`

- Source: `agbase`
- Verifier: `structured`
- Images: `data/raw/agbase/full/831976/831976_2.jpg`, `data/raw/agbase/full/831976/831976_1.jpg`

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
Diagnosis: sumac, japanese knotweed
Evidence: Visible symptoms should be checked against the image and crop context.
Uncertainty: Image-only assessment; confirm with field context before treatment.
Management: for japanese knotweed, refer to control methods outlined in https://extension.psu.edu/japanese-knotweed. for sumac, refer to mechanical control and chemical control methods outlined in https://extension.umn.edu/planting-and-growing-guides/woody-vegetation-control#chemical-control-1480161. dispose of any tree or plant cuttings properly to avoid accidental infestation. burning is an option for disposal. removal may take more than one year to accomplish.
Follow-up: Share close-up images and crop history if symptoms progress or the diagnosis is uncertain.
```

### consultation.32 `mirage-#434713`

- Source: `mirage`
- Verifier: `exact_match`
- Images: `data/raw/mirage/full/images/MMST_Standard/train/MMST_Standard-train-004298-01.png`, `data/raw/mirage/full/images/MMST_Standard/train/MMST_Standard-train-004298-02.png`

**Prompt**

```text
Please answer the following user question. You should first analyze the provided image, mentioning any visible clues or observations. Then, present the identification result. Write the entire content as one coherent paragraph (analysis + results).
User: What is the name of this plant? I think it is a fern?

Respond using these line-start section headers exactly once:
Diagnosis:
Evidence:
Uncertainty:
Management:
Follow-up:
```

**Target**

```text
Diagnosis: The images display leafy fronds with pinnately divided leaflets that are typical of ferns. The leaflets have a shiny green surface and appear to be attached to a long central stem or rachis. While the photos do not clearly show the leaf undersides, the characteristic segmented frond structure and the slight curling of some leaflets at the edges strongly suggest a fern. The expert's note on checking for sporangia—which are spore-producing structures usually found on the underside of fern leaves—is an excellent way to confirm this identification. Given these observations, this plant is indeed a fern, likely belonging to the genus Luzula ×somedana, a recognized fern species.
Evidence: Visible symptoms should be checked against the image and crop context.
Uncertainty: Image-only assessment; confirm with field context before treatment.
Management: No source-specific management step was provided; recommend local extension follow-up.
Follow-up: Share close-up images and crop history if symptoms progress or the diagnosis is uncertain.
```


## vqa

### vqa.1 `plantvillage_vqa-image_027506.JPG-080839`

- Source: `plantvillage_vqa`
- Verifier: `exact_match`
- Images: `data/raw/plantvillage_vqa/full/images/train/image_027506.JPG`

**Prompt**

```text
Does this specimen display perfect health characteristics?

Respond in this format:
Answer: <Yes or No>
```

**Target**

```text
Answer: No
```

### vqa.2 `plantvillage_vqa-image_023348.JPG-068462`

- Source: `plantvillage_vqa`
- Verifier: `exact_match`
- Images: `data/raw/plantvillage_vqa/full/images/train/image_023348.JPG`

**Prompt**

```text
What is the scientific identification of this plant sample?

Respond in this format:
Answer: <short answer>
```

**Target**

```text
Answer: Orange
```

### vqa.3 `plantvillage_vqa-image_008096.JPG-023742`

- Source: `plantvillage_vqa`
- Verifier: `exact_match`
- Images: `data/raw/plantvillage_vqa/full/images/train/image_008096.JPG`

**Prompt**

```text
What pathological state is this plant experiencing?

Respond in this format:
Answer: <short answer>
```

**Target**

```text
Answer: Diagnosis: Huanglongbing (HLB), or citrus greening.
```

### vqa.4 `plantvillage_vqa-image_009866.JPG-177810`

- Source: `plantvillage_vqa`
- Verifier: `exact_match`
- Images: `data/raw/plantvillage_vqa/full/images/train/image_009866.JPG`

**Prompt**

```text
Are there dark ring patterns creating target-like formations?

Respond in this format:
Answer: <Yes or No>
```

**Target**

```text
Answer: No
```

### vqa.5 `plantvillage_vqa-image_028671.JPG-084247`

- Source: `plantvillage_vqa`
- Verifier: `exact_match`
- Images: `data/raw/plantvillage_vqa/full/images/train/image_028671.JPG`

**Prompt**

```text
Does this leaf belong to a tomato plant species?

Respond in this format:
Answer: <Yes or No>
```

**Target**

```text
Answer: Yes
```

### vqa.6 `plantvillage_vqa-image_001848.JPG-005428`

- Source: `plantvillage_vqa`
- Verifier: `exact_match`
- Images: `data/raw/plantvillage_vqa/full/images/train/image_001848.JPG`

**Prompt**

```text
Does this plant exhibit peak physiological condition?

Respond in this format:
Answer: <Yes or No>
```

**Target**

```text
Answer: No
```

### vqa.7 `plantvillage_vqa-image_003457.JPG-009983`

- Source: `plantvillage_vqa`
- Verifier: `exact_match`
- Images: `data/raw/plantvillage_vqa/full/images/train/image_003457.JPG`

**Prompt**

```text
What is the cause of the unhealthy appearance of this Potato leaf?

Respond in this format:
Answer: <short answer>
```

**Target**

```text
Answer: These 'bullseye' lesions are a tell-tale sign of Early Blight.
```

### vqa.8 `plantvillage_vqa-image_023110.JPG-067765`

- Source: `plantvillage_vqa`
- Verifier: `exact_match`
- Images: `data/raw/plantvillage_vqa/full/images/train/image_023110.JPG`

**Prompt**

```text
Based on what's visible, what is the pathological diagnosis?

Respond in this format:
Answer: <short answer>
```

**Target**

```text
Answer: The large, dark, water-soaked lesions are a key sign of Late Blight.
```

### vqa.9 `plantvillage_vqa-image_030806.JPG-090704`

- Source: `plantvillage_vqa`
- Verifier: `exact_match`
- Images: `data/raw/plantvillage_vqa/full/images/train/image_030806.JPG`

**Prompt**

```text
Can you detect signs of pathogen presence on this foliage?

Respond in this format:
Answer: <Yes or No>
```

**Target**

```text
Answer: Yes
```

### vqa.10 `plantvillage_vqa-image_026826.JPG-078772`

- Source: `plantvillage_vqa`
- Verifier: `exact_match`
- Images: `data/raw/plantvillage_vqa/full/images/train/image_026826.JPG`

**Prompt**

```text
Can you identify any pathological conditions affecting this tomato foliage?

Respond in this format:
Answer: <short answer>
```

**Target**

```text
Answer: This is a viral infection: TYLCV.
```

### vqa.11 `plantvillage_vqa-image_008877.JPG-025935`

- Source: `plantvillage_vqa`
- Verifier: `exact_match`
- Images: `data/raw/plantvillage_vqa/full/images/train/image_008877.JPG`

**Prompt**

```text
Does this sample exhibit pathological irregularities?

Respond in this format:
Answer: <Yes or No>
```

**Target**

```text
Answer: Yes
```

### vqa.12 `plantvillage_vqa-image_022824.JPG-066909`

- Source: `plantvillage_vqa`
- Verifier: `exact_match`
- Images: `data/raw/plantvillage_vqa/full/images/train/image_022824.JPG`

**Prompt**

```text
What plant family does this specimen belong to?

Respond in this format:
Answer: <short answer>
```

**Target**

```text
Answer: Tomato
```

### vqa.13 `plantvillage_vqa-image_030508.JPG-089832`

- Source: `plantvillage_vqa`
- Verifier: `exact_match`
- Images: `data/raw/plantvillage_vqa/full/images/train/image_030508.JPG`

**Prompt**

```text
Can you determine what disease is impacting this tomato leaf sample?

Respond in this format:
Answer: <short answer>
```

**Target**

```text
Answer: The plant shows no symptoms of pathology.
```

### vqa.14 `plantvillage_vqa-image_003092.JPG-008964`

- Source: `plantvillage_vqa`
- Verifier: `exact_match`
- Images: `data/raw/plantvillage_vqa/full/images/train/image_003092.JPG`

**Prompt**

```text
Can you specify the genus and species of this vegetation?

Respond in this format:
Answer: <short answer>
```

**Target**

```text
Answer: Orange
```

### vqa.15 `plantvillage_vqa-image_006364.JPG-018551`

- Source: `plantvillage_vqa`
- Verifier: `exact_match`
- Images: `data/raw/plantvillage_vqa/full/images/train/image_006364.JPG`

**Prompt**

```text
Please give the official disease classification for this foliage.

Respond in this format:
Answer: <short answer>
```

**Target**

```text
Answer: This is Bacterial Spot. Note the small, dark, water-soaked lesions.
```

### vqa.16 `plantvillage_vqa-image_039845.JPG-117225`

- Source: `plantvillage_vqa`
- Verifier: `exact_match`
- Images: `data/raw/plantvillage_vqa/full/images/train/image_039845.JPG`

**Prompt**

```text
Which causal agents are responsible for this tomato leaf's abnormal appearance?

Respond in this format:
Answer: <short answer>
```

**Target**

```text
Answer: The symptoms, especially the olive-green mold on the leaf underside, indicate Leaf Mold.
```

### vqa.17 `plantvillage_vqa-image_015100.JPG-044187`

- Source: `plantvillage_vqa`
- Verifier: `exact_match`
- Images: `data/raw/plantvillage_vqa/full/images/train/image_015100.JPG`

**Prompt**

```text
Can you identify the origin of these visible defects?

Respond in this format:
Answer: <short answer>
```

**Target**

```text
Answer: The large, dark, water-soaked lesions are a key sign of Late Blight.
```

### vqa.18 `plantvillage_vqa-image_026306.JPG-077294`

- Source: `plantvillage_vqa`
- Verifier: `exact_match`
- Images: `data/raw/plantvillage_vqa/full/images/train/image_026306.JPG`

**Prompt**

```text
What symptoms might appear if this Tomato were infected with Target Spot?

Respond in this format:
Answer: <short answer>
```

**Target**

```text
Answer: A healthy version of this plant would not exhibit the dark, saturated spots with chlorotic borders.
```

### vqa.19 `plantvillage_vqa-image_030562.JPG-089995`

- Source: `plantvillage_vqa`
- Verifier: `exact_match`
- Images: `data/raw/plantvillage_vqa/full/images/train/image_030562.JPG`

**Prompt**

```text
Specify the scientific name of this plant variety.

Respond in this format:
Answer: <short answer>
```

**Target**

```text
Answer: Tomato
```

### vqa.20 `plantvillage_vqa-image_015603.JPG-045638`

- Source: `plantvillage_vqa`
- Verifier: `exact_match`
- Images: `data/raw/plantvillage_vqa/full/images/train/image_015603.JPG`

**Prompt**

```text
Would you deliver a comprehensive visual analysis of this plant specimen?

Respond in this format:
Answer: <short answer>
```

**Target**

```text
Answer: The cause is the bacterium Candidatus Liberibacter asiaticus, leading to Huanglongbing.
```

### vqa.21 `plantvillage_vqa-image_020890.JPG-061096`

- Source: `plantvillage_vqa`
- Verifier: `exact_match`
- Images: `data/raw/plantvillage_vqa/full/images/train/image_020890.JPG`

**Prompt**

```text
Does this leaf display symptoms indicative of infection?

Respond in this format:
Answer: <Yes or No>
```

**Target**

```text
Answer: No
```

### vqa.22 `plantvillage_vqa-image_033480.JPG-098494`

- Source: `plantvillage_vqa`
- Verifier: `exact_match`
- Images: `data/raw/plantvillage_vqa/full/images/train/image_033480.JPG`

**Prompt**

```text
What root factor creates these observable symptoms?

Respond in this format:
Answer: <short answer>
```

**Target**

```text
Answer: The causal agent is the Tomato Yellow Leaf Curl Virus.
```

### vqa.23 `plantvillage_vqa-image_007757.JPG-022734`

- Source: `plantvillage_vqa`
- Verifier: `exact_match`
- Images: `data/raw/plantvillage_vqa/full/images/train/image_007757.JPG`

**Prompt**

```text
Does this leaf belong to an orange tree specimen?

Respond in this format:
Answer: <Yes or No>
```

**Target**

```text
Answer: Yes
```

### vqa.24 `plantvillage_vqa-image_004272.JPG-012417`

- Source: `plantvillage_vqa`
- Verifier: `exact_match`
- Images: `data/raw/plantvillage_vqa/full/images/train/image_004272.JPG`

**Prompt**

```text
What plant family does this specimen belong to?

Respond in this format:
Answer: <short answer>
```

**Target**

```text
Answer: Corn
```

### vqa.25 `plantvillage_vqa-image_024821.JPG-072732`

- Source: `plantvillage_vqa`
- Verifier: `exact_match`
- Images: `data/raw/plantvillage_vqa/full/images/train/image_024821.JPG`

**Prompt**

```text
Which distinguishing features and visible traits characterize this botanical sample?

Respond in this format:
Answer: <short answer>
```

**Target**

```text
Answer: The diagnosis is Tomato Yellow Leaf Curl Virus (TYLCV).
```

### vqa.26 `plantvillage_vqa-image_013336.JPG-039017`

- Source: `plantvillage_vqa`
- Verifier: `exact_match`
- Images: `data/raw/plantvillage_vqa/full/images/train/image_013336.JPG`

**Prompt**

```text
Guide me through the foliage conditions and features in this picture.

Respond in this format:
Answer: <short answer>
```

**Target**

```text
Answer: The plant is affected by Bacterial Spot disease.
```

### vqa.27 `plantvillage_vqa-image_027674.JPG-081267`

- Source: `plantvillage_vqa`
- Verifier: `exact_match`
- Images: `data/raw/plantvillage_vqa/full/images/train/image_027674.JPG`

**Prompt**

```text
Present a detailed analysis of the specimen's visual attributes and wellness factors.

Respond in this format:
Answer: <short answer>
```

**Target**

```text
Answer: The cause is the bacterium Candidatus Liberibacter asiaticus, leading to Huanglongbing.
```

### vqa.28 `plantvillage_vqa-image_010589.JPG-030868`

- Source: `plantvillage_vqa`
- Verifier: `exact_match`
- Images: `data/raw/plantvillage_vqa/full/images/train/image_010589.JPG`

**Prompt**

```text
What pathological state is this plant experiencing?

Respond in this format:
Answer: <short answer>
```

**Target**

```text
Answer: The fungus Alternaria solani, which causes Early Blight, is responsible.
```

### vqa.29 `plantvillage_vqa-image_025898.JPG-075969`

- Source: `plantvillage_vqa`
- Verifier: `exact_match`
- Images: `data/raw/plantvillage_vqa/full/images/train/image_025898.JPG`

**Prompt**

```text
What contrasting features distinguish this specimen from healthy reference points?

Respond in this format:
Answer: <short answer>
```

**Target**

```text
Answer: A healthy leaf would be uniformly green and free of the evidence of orange or brown pustular structures.
```

### vqa.30 `plantvillage_vqa-image_012134.JPG-035457`

- Source: `plantvillage_vqa`
- Verifier: `exact_match`
- Images: `data/raw/plantvillage_vqa/full/images/train/image_012134.JPG`

**Prompt**

```text
Does this leaf show evidence of pathogenic infection?

Respond in this format:
Answer: <Yes or No>
```

**Target**

```text
Answer: Yes
```

### vqa.31 `plantvillage_vqa-image_012876.JPG-184144`

- Source: `plantvillage_vqa`
- Verifier: `exact_match`
- Images: `data/raw/plantvillage_vqa/full/images/train/image_012876.JPG`

**Prompt**

```text
Can you detect yellowing, leaf curl patterns, and stunted development?

Respond in this format:
Answer: <Yes or No>
```

**Target**

```text
Answer: Yes
```

### vqa.32 `plantvillage_vqa-image_036008.JPG-169919`

- Source: `plantvillage_vqa`
- Verifier: `exact_match`
- Images: `data/raw/plantvillage_vqa/full/images/train/image_036008.JPG`

**Prompt**

```text
Does this Tomato leaf show signs of Tomato Yellow Leaf Curl Virus?

Respond in this format:
Answer: <Yes or No>
```

**Target**

```text
Answer: No
```

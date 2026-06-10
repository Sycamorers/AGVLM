# SFT Format Audit

- Manifest: `/blue/hmedeiros/qinruoyao/agvlm/data/manifests/full/sft_train_phi4_max3_balanced_v2_instructional_labelrepaired.jsonl`
- Prompt format: `instructional`
- Target format: `instructional`
- Unique examples rendered: `80`
- Validation failures: `0`

## Counts

| Task | Manifest rows | Rendered unique examples |
| --- | ---: | ---: |
| clarify_or_respond | 27000 | 20 |
| classification | 54000 | 20 |
| consultation | 36000 | 20 |
| vqa | 63000 | 20 |

## clarify_or_respond

### clarify_or_respond.1 `mirage-#838170`

- Source: `mirage`
- Verifier: `clarify`
- Images: `data/raw/mirage/full/images/MMMT_Direct/train/MMMT_Direct-train-000260-01.png`, `data/raw/mirage/full/images/MMMT_Direct/train/MMMT_Direct-train-000260-02.png`, `data/raw/mirage/full/images/MMMT_Direct/train/MMMT_Direct-train-000260-03.png`

**Prompt**

```text
user: Single sour cherry (Montmorency I think) tree in my back yard. About 7 years old. No spraying this year. Leaves are in good shape. Last year they had brown spots but not this year. For about 3 years now I've had these brown spots on the fruit. Starting shortly after the fruit got to about a quarter inch in diameter. About 85% of the fruit has these spots. Some brown rot (in photo) but very little.  Insect? (I don't see any) Fungus? Can you diagnose and tell me how to treat it? I would be very appreciative. I'd love to use them next year.
user: There seems to be more brown rot now. Still no leaves infected. Any Idea what this is?

Respond using exactly one of these formats:
Decision: clarify
Clarifying question: <one question needed before diagnosis or management>

Decision: respond
Answer: <concise agricultural answer>
```

**Target**

```text
Decision: respond
Answer: Based on the symptoms you've described and the images provided, it seems that your Montmorency sour cherry tree is primarily affected by brown rot, a common fungal disease caused by Monilinia species. The brown spots on the fruit and the progression to brown rot are typical signs. This disease can be exacerbated by wet conditions, which promote fungal growth. Since you mentioned that the leaves are unaffected, it is less likely to be a leaf-related fungal issue. Here are some steps you can take to manage and prevent brown rot in the future:

1. **Sanitation**: Remove and destroy any mummified or rotted fruit from the tree and the ground to reduce the source of fungal spores.
2. **Pruning**: Ensure good air circulation by pruning the tree to open up the canopy. This helps reduce humidity around the fruit.
3. **Fungicide Application**: Consider applying a fungicide labeled for use on cherry trees against brown rot. This should be done in early spring, starting at bloom and continuing at intervals as recommended on the product label.
4. **Bird Protection**: Since birds are picking the fruit, consider using netting to protect the cherries, which can also help reduce the spread of disease by preventing fruit damage.
5. **Monitor Weather Conditions**: Be vigilant during wet weather, as this can increase the risk of brown rot. Adjust your management practices accordingly.

Implementing these practices should help reduce the incidence of brown rot and improve the health and yield of your cherry tree next year.
```

### clarify_or_respond.2 `mirage-#836555`

- Source: `mirage`
- Verifier: `clarify`
- Images: `data/raw/mirage/full/images/MMMT_Direct/train/MMMT_Direct-train-002771-01.png`, `data/raw/mirage/full/images/MMMT_Direct/train/MMMT_Direct-train-002771-02.png`, `data/raw/mirage/full/images/MMMT_Direct/train/MMMT_Direct-train-002771-03.png`

**Prompt**

```text
user: Can you please confirm what type of larvae this is on my raspberry plants? Pictures of larvae and leaves included in hopes of positive identification. I’ve not been able to distinguish through the website.
Thank you!

Respond using exactly one of these formats:
Decision: clarify
Clarifying question: <one question needed before diagnosis or management>

Decision: respond
Answer: <concise agricultural answer>
```

**Target**

```text
Decision: clarify
Clarifying question: Could you please provide a description of the larvae, including any specific characteristics you have observed? Additionally, knowing your location or region can help narrow down the possibilities.
```

### clarify_or_respond.3 `mirage-#827461`

- Source: `mirage`
- Verifier: `clarify`
- Images: `data/raw/mirage/full/images/MMMT_Decomp/train/MMMT_Decomp-train-003005-01.png`, `data/raw/mirage/full/images/MMMT_Decomp/train/MMMT_Decomp-train-003005-02.png`, `data/raw/mirage/full/images/MMMT_Decomp/train/MMMT_Decomp-train-003005-03.png`

**Prompt**

```text
user: We have 10 river birch trees in our backyard. They are very sparse with leaves this year and have many dead branches. What do you recommend to improve the health these trees? We are located in Cypress.

Respond using exactly one of these formats:
Decision: clarify
Clarifying question: <one question needed before diagnosis or management>

Decision: respond
Answer: <concise agricultural answer>
```

**Target**

```text
Decision: clarify
Clarifying question: Could you provide more information on your watering schedule, soil conditions, and any fertilization practices you use? Additionally, have you noticed any signs of pests or diseases on the trees?
```

### clarify_or_respond.4 `mirage-#822298`

- Source: `mirage`
- Verifier: `clarify`
- Images: `data/raw/mirage/full/images/MMMT_Decomp/train/MMMT_Decomp-train-000738-01.png`

**Prompt**

```text
user: What is eating my roses? I don’t see aphids or evidence of caterpillars, larvae or any sort of mite. The flowers seem not to be affected, but the leaves look chewed , some just have brown spots and some of the older stems have black spots. Is it slugs/snails?

Respond using exactly one of these formats:
Decision: clarify
Clarifying question: <one question needed before diagnosis or management>

Decision: respond
Answer: <concise agricultural answer>
```

**Target**

```text
Decision: clarify
Clarifying question: Could you provide more information about the environmental conditions around your roses, such as humidity levels or recent weather changes? Also, have you noticed any slime trails that might indicate the presence of slugs or snails?
```

### clarify_or_respond.5 `mirage-#842288`

- Source: `mirage`
- Verifier: `clarify`
- Images: `data/raw/mirage/full/images/MMMT_Direct/train/MMMT_Direct-train-000120-01.png`, `data/raw/mirage/full/images/MMMT_Direct/train/MMMT_Direct-train-000120-02.png`, `data/raw/mirage/full/images/MMMT_Direct/train/MMMT_Direct-train-000120-03.png`

**Prompt**

```text
user: Hello, I have two Canadian Hemlocks off my patio and they look to be struggling, actually for years. They get new needles each spring but there seems to be a lot of needle drop and dead/sparse branches.  some are at the outside toward the top but also a lot interspersed. They are at least 30 years old. Pictures attached, hopefully enough to figure it out.  I have thrown some 10-10-10 granular fertilizer around the drip lines of all the trees this year.  thank you, Nancy
expert: Hi, Peyton, This looks like fabrella needle blight, which is a fungal disease that results in the loss of hemlock needles. The disease is not fatal unless the tree is under drought stress, which is likely not the case in your yard trees. You can read more about fabrella needle blight at the link below: https://elibrary.dcnr.pa.gov/PDFProvider.ashx?action=PDFStream&docID=1738101&chksum=&revision=0&docName=tHlthFactSheet-FabrellaNeedleBlight&nativeExt=pdf  Sanitation is the recommended practice to limit the effect of the disease. Gather up fallen needles each spring. Trim off affected branches during winter months to increase air flow, which will also limit moist conditions in which fungal diseases can thrive. Keep trees watered in times of drought. The annual application of fertilizer in the fall or spring months is also a good idea.  If you have any questions, please feel free to reach out.  Thank you, -Julie
user: Thank you. What type of fertilizer would you recommend?  Picking up these tiny needles from the ground beneath would be virtually impossible.  The area has some mulch and sits in a garden bed with other plants.  Are there any recommended sprays or just the course of treatment you mention below? Thank you, Peyton: ask=<personal data hidden> <ask=<personal data hidden>> On Behalf Of Ask Extension Sent: Wednesday, July 26, 2023 10:22 AM To: Nancy E Vogrin <<personal data hidden>> Subject: Re: Canadian Hemlock issue (#0112214)
expert: Hi, Peyton, A standard fertilizer applied in the fall is fine. 10-10-10 is very much acceptable.  I understand it may be difficult to gather the needles. Perhaps use a blower to remove them from under the tree? No chemical application is recommended. Trimming out the affected branches will help.  Thank you, -Julie
user: Thank you, Elliot!   From: ask=<personal data hidden> <ask=<personal data hidden>> On Behalf Of Ask Extension Sent: Wednesday, July 26, 2023 12:08 PM To: Nancy E Vogrin <<personal data hidden>> Subject: Re: Canadian Hemlock issue (#0112214)

Respond using exactly one of these formats:
Decision: clarify
Clarifying question: <one question needed before diagnosis or management>

Decision: respond
Answer: <concise agricultural answer>
```

**Target**

```text
Decision: respond
Answer: Hi, Peyton, when dealing with fabrella needle blight, it's important to balance the health of the tree with maintaining its structure. You should prioritize removing branches that are completely dead, as they no longer contribute to the tree's health and can harbor disease. For branches that are affected but still have some healthy needles, consider trimming only the most severely affected parts to improve air circulation and reduce disease spread, while preserving as much of the healthy foliage as possible. This approach helps maintain the tree's overall health and appearance. If the tree is significantly sparse, it might be beneficial to consult with a local arborist for a more tailored approach. Thank you, -Julie
```

### clarify_or_respond.6 `mirage-#874261`

- Source: `mirage`
- Verifier: `clarify`
- Images: `data/raw/mirage/full/images/MMMT_Decomp/train/MMMT_Decomp-train-001543-01.png`, `data/raw/mirage/full/images/MMMT_Decomp/train/MMMT_Decomp-train-001543-02.png`

**Prompt**

```text
user: I have a Hibiscus laevis in my garden (planted spring 2023). It flowered last year and came up again this year but the leaves are getting destroyed by something. Any ideas about what could be causing the harm and how to treat?

Respond using exactly one of these formats:
Decision: clarify
Clarifying question: <one question needed before diagnosis or management>

Decision: respond
Answer: <concise agricultural answer>
```

**Target**

```text
Decision: clarify
Clarifying question: Could you describe the type of damage on the leaves? Are there holes, discoloration, or any visible pests? Also, could you provide some details about the recent weather conditions and your watering schedule? If possible, please describe what is shown in the attached images.
```

### clarify_or_respond.7 `mirage-#874660`

- Source: `mirage`
- Verifier: `clarify`
- Images: `data/raw/mirage/full/images/MMMT_Decomp/train/MMMT_Decomp-train-000216-01.png`, `data/raw/mirage/full/images/MMMT_Decomp/train/MMMT_Decomp-train-000216-02.png`

**Prompt**

```text
user: First, let me thank you for this service as I greatly appreciate any help you can provide to me.  I had a new lawn sodded with blue grass last fall everything was beautiful until a few weeks ago when I noticed these yellowish spots in my yard. I am located in Austin, MN, but the lawn I am referring to is just south of Hayward, Wisconsin.
Upon doing some research it appears to be poa trivialis. I would like to treat the weeds so my lawn can return to its original beautiful state.  I am having trouble finding a solution.  Any help you can provide is appreciated.  Thank you again for this service!
expert: I wish I had a good answer for you. Pop Trivialis is difficult to get rid of.  How extensive it is will probably determine your course of action. You can try to pull it if it is not too extensive. It is intolerant of dry soil so it may be easier to get at this summer if we reach a dry spell.  Do not over irrigate your lawn.  If you choose to use an herbicide you will have to reseed the areas because it will kill all the grass in the treated area.  Are you certain of your identification? One of your photos shows seed heads which is more likely to be poa annua. If this is the case you can use a pre-emergent next year.  https://plants.ces.ncsu.edu/plants/poa-annua/ https://apps.extension.umn.edu/garden/diagnose/weed/grass/roughbluegrass.html https://extension.psu.edu/lawn-and-turfgrass-weeds-roughstalk-bluegrass-poa-trivialis-l https://turf.purdue.edu/poa-annua-poa-trivialis-in-lawns-and-athletic-fields/
user: Do you feel that Acclaim Extra Selective Herbicide is a good choice to treat it?      On Thu, Jun 27, 2024 at 10:59 AM Ask Extension <<personal data hidden>> wrote:

Respond using exactly one of these formats:
Decision: clarify
Clarifying question: <one question needed before diagnosis or management>

Decision: respond
Answer: <concise agricultural answer>
```

**Target**

```text
Decision: clarify
Clarifying question: Can you confirm if the identification of Poa trivialis is accurate, and how extensive is the infestation in your lawn? Additionally, could you provide details on your current lawn care practices, such as irrigation and mowing frequency?
```

### clarify_or_respond.8 `mirage-#842177`

- Source: `mirage`
- Verifier: `clarify`
- Images: `data/raw/mirage/full/images/MMMT_Decomp/train/MMMT_Decomp-train-000553-01.png`, `data/raw/mirage/full/images/MMMT_Decomp/train/MMMT_Decomp-train-000553-02.png`, `data/raw/mirage/full/images/MMMT_Decomp/train/MMMT_Decomp-train-000553-03.png`

**Prompt**

```text
user: I have a red bug that is killing my plants. I also saw it on my walk to nearby Willow Pond. I took a photo at that time and I'll include it(can't include as I can only upload 3) as well as the ones I took today 7/23/23. It's in this plant that I don't know the name of and I saw them in one of my black eyed susans as well.

How do I get rid of this bug?

Thanks, Charlie

Respond using exactly one of these formats:
Decision: clarify
Clarifying question: <one question needed before diagnosis or management>

Decision: respond
Answer: <concise agricultural answer>
```

**Target**

```text
Decision: respond
Answer: It sounds like you might be dealing with aphids, which are often small and can appear red. Spraying them off with a strong jet of water is a good initial step, as it can physically remove them from the plants. Aphids are unlikely to drown, but dislodging them can reduce their numbers significantly. Consistent spraying can help manage the population, but for a more long-term solution, you might consider introducing natural predators like ladybugs or using insecticidal soap. Ensure to monitor your plants regularly for any resurgence of the pests.
```

### clarify_or_respond.9 `mirage-#826473`

- Source: `mirage`
- Verifier: `clarify`
- Images: `data/raw/mirage/full/images/MMMT_Decomp/train/MMMT_Decomp-train-003234-01.png`

**Prompt**

```text
user: I have had a large bed of mahonia in my garden for almost 20 years. It has been so hardy and beautiful.  This year almost every leaf is brown or bronze and most have dropped off, for seemingly no reason. Other beds of mahonia are fine. This faces south, and is under a large tree. Should I cut all the stalks back and hope for regrowth? Thanks for any help you can give me! It looks terrible! jc
expert: JC,  Mahonia is a fairly hardy plant.  Since you have other mahonias that are doing well I would wonder what happened in this location to affect this shrub.  Was there more sun exposure last year?  Did you thin or prune the tree it is growing under?  Was the amount of water getting to this shrub less than the other Mahonias?  Although, they are drought tolerant, they don't tolerate hot winds or high heat.   I'd recommend a hard pruning to see if it comes back more healthy.  Are there any flowers?  Recommended pruning time for mahonia is after flowering in the spring.    They can also get "rust", but  I don't see rust spots in the photo you sent   Here are some resources  http://coloradomountaingardener.blogspot.com/2018/06/a-native-in-garden-oregon-grape.html
user: Thank you so much for your answer. The tree above the mahonia lost one limb but I don't think it would affect the light that much. Everything else is the same as normal, as far as water etc. There are no flowers this year, so should I wait to prune or do it now? Should I also clear out all the dead leaves? You are an enormous resource. Thank you againjudithOn Sat, Apr 22, 2023 at 12:48 PM Ask Extension <<personal data hidden>> wrote:

Respond using exactly one of these formats:
Decision: clarify
Clarifying question: <one question needed before diagnosis or management>

Decision: respond
Answer: <concise agricultural answer>
```

**Target**

```text
Decision: respond
Answer: Since there are no flowers this year, you can go ahead and prune the mahonia now. Pruning can help stimulate new growth and improve the plant's health. Additionally, clearing out the dead leaves is a good idea as it can help prevent any potential disease and improve air circulation around the plant. Keep an eye on the plant's recovery and ensure it receives adequate water, especially during dry spells. If the problem persists, consider consulting a local expert to assess any environmental factors that might be affecting the plant.
```

### clarify_or_respond.10 `mirage-#885196`

- Source: `mirage`
- Verifier: `clarify`
- Images: `data/raw/mirage/full/images/MMMT_Direct/train/MMMT_Direct-train-003462-01.png`, `data/raw/mirage/full/images/MMMT_Direct/train/MMMT_Direct-train-003462-02.png`

**Prompt**

```text
user: I’ve been noticing these issues on my Japanese Maple leaves for a while now and not sure if it’s due to warmer temperatures and/or lack of water or a disease like powdery mildew.
expert: Hi, Peyton is typical on lave leaf plants at the end of the summer, especially with the high heat and drought conditions we had this past season.  Did you supplement irrigation this summer? If not, it would be a good idea to water periodically this fall to help it into winter. We are supposed to get some rain this week, but it has been a pretty dry month so far. You could give it a good soak today by laying a garden hose on a slow trickle at the base for about 20 minutes.  You can review out Watering Trees and Shrubs page for further guidance.  The leaves will fall off this autumn and should be fine next spring.  Emily
user: I have irrigation for my yard, but I’m sure I probably could’ve watered my tree a lot more than I have. Do those type of trees require a lot of water? I wasn’t sure if I only need to use a water pitcher to be enough or if normally it needs a good hose soaking like you mentioned. ----------------------------------- Jason

Respond using exactly one of these formats:
Decision: clarify
Clarifying question: <one question needed before diagnosis or management>

Decision: respond
Answer: <concise agricultural answer>
```

**Target**

```text
Decision: respond
Answer: Japanese Maple trees generally prefer consistent moisture, especially during hot and dry periods. While your lawn irrigation might not be sufficient for the tree, using a hose to water deeply is a good approach. Ideally, you should aim to water the tree deeply once a week, especially during dry spells. This means letting the hose run at a slow trickle at the base of the tree for about 20-30 minutes, ensuring the water penetrates deeply into the soil. This encourages deep root growth and helps the tree withstand drought conditions. Using a watering can for supplemental watering is fine, but it may not provide enough water for deep soil penetration. Monitoring the soil moisture by checking a few inches below the surface can also help you determine when to water. Additionally, applying a layer of mulch around the base of the tree can help retain soil moisture and regulate temperature. If you continue to notice issues with the leaves, it might be worth considering other factors such as soil quality or potential diseases, but proper watering is a good first step.
```

### clarify_or_respond.11 `mirage-#831742`

- Source: `mirage`
- Verifier: `clarify`
- Images: `data/raw/mirage/full/images/MMMT_Decomp/train/MMMT_Decomp-train-002449-01.png`

**Prompt**

```text
user: What are these red flowering plants?  Thanks!
expert: Hi, Taylor- Those are bread poppies or opium poppies from which come poppy seeds and... well... opium. They are illegal to plant in several states but don't know about Oregon. Hard to find the seeds though. https://museum.dea.gov/exhibits/online-exhibits/cannabis-coca-and-poppy-natures-addictive-plants/opium-poppy
user: Thanks, Skyler!  That’ll what I told my husband, but he wanted to hear from om an expert!Kris On Wed, May 24, 2023 at 1:51 PM Ask Extension <<personal data hidden>> wrote:

Respond using exactly one of these formats:
Decision: clarify
Clarifying question: <one question needed before diagnosis or management>

Decision: respond
Answer: <concise agricultural answer>
```

**Target**

```text
Decision: respond
Answer: You're welcome, Kris! It's always good to confirm plant identifications, especially with something as unique as poppies. If you have any more questions about your garden or need further assistance, feel free to ask!
```

### clarify_or_respond.12 `mirage-#825399`

- Source: `mirage`
- Verifier: `clarify`
- Images: `data/raw/mirage/full/images/MMMT_Decomp/train/MMMT_Decomp-train-003004-01.png`, `data/raw/mirage/full/images/MMMT_Decomp/train/MMMT_Decomp-train-003004-02.png`, `data/raw/mirage/full/images/MMMT_Decomp/train/MMMT_Decomp-train-003004-03.png`

**Prompt**

```text
user: To avoid a repeat of antler rub damage that occurred in the fall of 2021, I put a protective sleeve around my young Freeman's Maple in October 2022. I used a length of 4" diameter corrugated black drain tile, slit up the side. I just got around to removing the sleeve last week, and noticed some strange damage near the base of the tree. Looks like something small chewed several holes through the bark. There is a section of bark that feels like it might be hollow underneath - flexes a bit when I push on it.

Respond using exactly one of these formats:
Decision: clarify
Clarifying question: <one question needed before diagnosis or management>

Decision: respond
Answer: <concise agricultural answer>
```

**Target**

```text
Decision: respond
Answer: The damage you're describing, with small holes and a section of bark that feels hollow, is likely caused by rodents or small mammals that found the area under the protective sleeve to be a safe place to nest or feed. The hollow feeling under the bark suggests that the cambium layer might have been damaged, which can affect the tree's ability to transport nutrients.

To address this, first, carefully inspect the damaged area to assess the extent of the damage. If the bark is loose and the cambium layer is exposed, it may be necessary to trim away any dead or loose bark to prevent further decay and allow the tree to heal. Ensure that the area is kept clean and dry to prevent fungal infections.

In the future, consider using a different type of protective sleeve that allows for better air circulation and doesn't provide a habitat for pests. Additionally, check the sleeve periodically throughout the season to ensure no damage is occurring underneath. If the damage is extensive, consulting with a local arborist for an in-person assessment might be beneficial.
```

### clarify_or_respond.13 `mirage-#832536`

- Source: `mirage`
- Verifier: `clarify`
- Images: `data/raw/mirage/full/images/MMMT_Direct/train/MMMT_Direct-train-003729-01.png`, `data/raw/mirage/full/images/MMMT_Direct/train/MMMT_Direct-train-003729-02.png`

**Prompt**

```text
user: Just planted a comice pear tree last October.  Now the leaves have some white spots on them, are curled, some holes and some dark leaves.  We had a comice tree in the same location and it did fine for several years, then it got dark leaves and dead branches.  Pictures attached.  We have a bartlett close by and it is doing fine.  Thanks for your help.
expert: Hello.  From the pictures, it appears your have pear leaf blister as defined below:     Cause The fungus Taphrina bullata has been reported from Washington. Not a common issue in commercial production of pears likely due to pest control measures targeting other disease and insects. Symptoms Large, light-green to yellow, irregular blisters form on the upper surface of leaves. A white fungal bloom develops on the concave underside. The leaf blisters caused by eriophyid mites (pear rust mite, Epitrimerus pyri) are more common, smaller, can be reddish and are numerous on the leaf surface.   Please confirm if these are the symptoms you are seeing.  I will leave this question open for you to reply and let me know if this sounds right or not
user: It doesn't quite sound right.  On the underside of the leaves, there are brown spots.  On the upper side, they are more like yellow. They also have some holes in the leaves like an insect has eaten them.  A few of the leaves turned brown.  Not sure.  IT could be what you described.Can the tree be saved?  How do we prevent this in the future?  An anitfungal?  Can this spread to our bartlett which is in close proximity?Thanks for your help.  On Wednesday, May 31, 2023 at 07:58:59 AM PDT, Ask Extension <<personal data hidden>> wrote:

Respond using exactly one of these formats:
Decision: clarify
Clarifying question: <one question needed before diagnosis or management>

Decision: respond
Answer: <concise agricultural answer>
```

**Target**

```text
Decision: respond
Answer: Based on your description and the presence of juniper trees nearby, it does sound like your comice pear tree might be affected by pear rust, which is a fungal disease that can be exacerbated by the presence of junipers. To address this issue, you can apply a fungicide that is effective against rust diseases. In southern Oregon, products containing myclobutanil or propiconazole are often recommended. It's best to apply the fungicide in early spring before the symptoms appear, but since your tree is already showing symptoms, you can still apply it now to help manage the disease. Be sure to follow the label instructions for application rates and safety precautions.

To prevent future infections, consider the following steps:
1. Remove any fallen leaves and debris around the tree to reduce sources of infection.
2. Prune the tree to improve air circulation, which can help reduce humidity and fungal growth.
3. Monitor the tree regularly for early signs of disease and treat promptly.

Regarding your Bartlett pear tree, it is possible for rust to spread, so keep an eye on it and consider applying a preventative fungicide treatment as well. Unfortunately, removing the junipers may not be feasible, but managing the disease on your pear trees can help mitigate the impact.

I hope this helps, and please feel free to reach out if you have any more questions or need further assistance.
```

### clarify_or_respond.14 `mirage-#840944`

- Source: `mirage`
- Verifier: `clarify`
- Images: `data/raw/mirage/full/images/MMMT_Direct/train/MMMT_Direct-train-001653-01.png`, `data/raw/mirage/full/images/MMMT_Direct/train/MMMT_Direct-train-001653-02.png`, `data/raw/mirage/full/images/MMMT_Direct/train/MMMT_Direct-train-001653-03.png`

**Prompt**

```text
user: Hello, we live in Rockville. We have had arbor vitae in our backyard for over ten years. I spray them several times of year with deer repellant and we have been spared the deer eating the lower limbs. We were surprised this weekend to find a hole in the top of one of the trees as if something had ripped out some of the branches. The rest of the tree is healthy as are the other trees.
Do you have any thoughts as to what might have caused this damage?
Anything we need to do to enhance growth to repair the defect?
Thanks,

Respond using exactly one of these formats:
Decision: clarify
Clarifying question: <one question needed before diagnosis or management>

Decision: respond
Answer: <concise agricultural answer>
```

**Target**

```text
Decision: clarify
Clarifying question: Could you provide more details about the recent weather conditions or any maintenance activities that might have occurred recently? Additionally, have you noticed any signs of pests or animals other than deer in the area?
```

### clarify_or_respond.15 `mirage-#835546`

- Source: `mirage`
- Verifier: `clarify`
- Images: `data/raw/mirage/full/images/MMMT_Decomp/train/MMMT_Decomp-train-002860-01.png`

**Prompt**

```text
user: I believe I have asian lady beetle larvae on my hydrangeas. I know they eat aphids but not sure if I should destroy them or are they ok to leave alone?  I only see a few here and there on my 7 brushes.

Respond using exactly one of these formats:
Decision: clarify
Clarifying question: <one question needed before diagnosis or management>

Decision: respond
Answer: <concise agricultural answer>
```

**Target**

```text
Decision: respond
Answer: Asian lady beetles are indeed known to be more aggressive than native ladybugs and can outcompete them. While they do help control aphid populations, their invasive nature and potential to harm native species are valid concerns. If you are seeing only a few larvae and they are not causing significant harm to your plants, you might choose to leave them for now. However, if you are concerned about their impact on native ladybug populations, you could consider removing them manually. It's a balance between managing pest populations and preserving native species.
```

### clarify_or_respond.16 `mirage-#884761`

- Source: `mirage`
- Verifier: `clarify`
- Images: `data/raw/mirage/full/images/MMMT_Direct/train/MMMT_Direct-train-002974-01.png`, `data/raw/mirage/full/images/MMMT_Direct/train/MMMT_Direct-train-002974-02.png`, `data/raw/mirage/full/images/MMMT_Direct/train/MMMT_Direct-train-002974-03.png`

**Prompt**

```text
user: 1.) May I cut down our taller Grases (attached) in the backgroud to 6 inches prior to winter and expect them to return with nice growth next season?

2.) Regarding attached pics of our 25 year-old Cherry Tree in our memorial garden, our cherry tree (25 years old) is leaking what appears to be a molasses-like substance. Are you able to provide a perspective on our cherry tree, as to its health?

Thank you for your time and expertise.

Respond using exactly one of these formats:
Decision: clarify
Clarifying question: <one question needed before diagnosis or management>

Decision: respond
Answer: <concise agricultural answer>
```

**Target**

```text
Decision: clarify
Clarifying question: Could you please specify the type of grasses you have and your location or climate zone? Additionally, are there any other symptoms or changes in the cherry tree besides the leaking substance?
```

### clarify_or_respond.17 `mirage-#828607`

- Source: `mirage`
- Verifier: `clarify`
- Images: `data/raw/mirage/full/images/MMMT_Direct/train/MMMT_Direct-train-002153-01.png`, `data/raw/mirage/full/images/MMMT_Direct/train/MMMT_Direct-train-002153-02.png`, `data/raw/mirage/full/images/MMMT_Direct/train/MMMT_Direct-train-002153-03.png`

**Prompt**

```text
user: Included pics of some tomato seedlings… they were potted up approx 1 wk ago - mix is coconut coir & perlite (2 to 1) then I bottom watered them w 1/2 strength Neptune’s Harvest tomato & vege .. now I see several discolored leaves- what is the issue & what do I do to correct?
Thank you.

Respond using exactly one of these formats:
Decision: clarify
Clarifying question: <one question needed before diagnosis or management>

Decision: respond
Answer: <concise agricultural answer>
```

**Target**

```text
Decision: clarify
Clarifying question: Could you describe the discoloration in more detail? Additionally, what are the current environmental conditions like (light, temperature, humidity), and how often are you watering the seedlings? Have there been any recent changes in their care?
```

### clarify_or_respond.18 `mirage-#878069`

- Source: `mirage`
- Verifier: `clarify`
- Images: `data/raw/mirage/full/images/MMMT_Decomp/train/MMMT_Decomp-train-002376-01.png`

**Prompt**

```text
user: Why does this hydrangea not open up the bloom?  There are a couple per cluster, but not big blooms.
expert: Thank you for an interesting question, but before I can provide any ideas I need a little more information.     Has the plant bloomed previously?   How old is the plant and have you done any pruning on it?     If you could give me this information I may be able to provide some helpful suggestions.   Thank you.
user: Hello-  This has never fully bloomed, it gets the small clusters and a couple will look like a small flower but not many.  I got a cutting from a friend and planted it in 2019.  I have never pruned it.  Thank you- Joan Fisk (she/her/hers) | Administrative Assistant | Smith Elementary  Phone: <personal data hidden> | Fax: <personal data hidden> | Attendance:<personal data hidden>   "Diversity is being invited to the party;Inclusion is being asked to dance."                                  -Verna Myers      On Mon, Jul 22, 2024 at 10:59 AM Ask Extension <<personal data hidden>> wrote:

Respond using exactly one of these formats:
Decision: clarify
Clarifying question: <one question needed before diagnosis or management>

Decision: respond
Answer: <concise agricultural answer>
```

**Target**

```text
Decision: respond
Answer: Thank you for providing more details, Joan. Since your hydrangea is possibly an Annabelle variety, which typically blooms well, there are a few factors we can consider to help improve its blooming. Firstly, Annabelle hydrangeas can benefit from regular fertilization, especially if the soil is not very rich. Consider using a balanced, slow-release fertilizer in early spring. Secondly, while Annabelle hydrangeas can tolerate full sun, they often perform better with some afternoon shade, especially in hotter climates. If possible, try to provide some shade during the hottest part of the day. Lastly, ensure the plant is receiving adequate water, as hydrangeas prefer consistently moist soil. You might also consider pruning the plant in late winter or early spring to encourage new growth, which can lead to more blooms. I hope these suggestions help your hydrangea thrive!
```

### clarify_or_respond.19 `mirage-#836476`

- Source: `mirage`
- Verifier: `clarify`
- Images: `data/raw/mirage/full/images/MMMT_Direct/train/MMMT_Direct-train-001042-01.png`, `data/raw/mirage/full/images/MMMT_Direct/train/MMMT_Direct-train-001042-02.png`, `data/raw/mirage/full/images/MMMT_Direct/train/MMMT_Direct-train-001042-03.png`

**Prompt**

```text
user: Our thula after 3 years is starting to turn brown. Not sure why. The others are doing great. Would love to save it, since it’s taken forever to grown and height.

Respond using exactly one of these formats:
Decision: clarify
Clarifying question: <one question needed before diagnosis or management>

Decision: respond
Answer: <concise agricultural answer>
```

**Target**

```text
Decision: clarify
Clarifying question: Could you provide more details about the environmental conditions such as sunlight exposure, watering schedule, soil type, and any recent changes in care or environment? Additionally, have you noticed any signs of pests or diseases on the affected plant?
```

### clarify_or_respond.20 `mirage-#873725`

- Source: `mirage`
- Verifier: `clarify`
- Images: `data/raw/mirage/full/images/MMMT_Decomp/train/MMMT_Decomp-train-002543-01.png`

**Prompt**

```text
user: What is this bug?  I found several of them around my bean and pea seedlings in the early evening that I assume have been destroyed by these bugs.   They probably have infested my potato plants as well.

Respond using exactly one of these formats:
Decision: clarify
Clarifying question: <one question needed before diagnosis or management>

Decision: respond
Answer: <concise agricultural answer>
```

**Target**

```text
Decision: clarify
Clarifying question: Could you provide more details about the size, shape, and behavior of the bugs you observed? Additionally, have you tried any pest control measures so far?
```


## classification

### classification.1 `plantvillage-train-037631`

- Source: `plantvillage`
- Verifier: `label`
- Images: `data/raw/plantvillage/full/images/train/037631.png`

**Prompt**

```text
Identify the crop issue or pest in this agricultural image.

Respond in this format:
Answer: <canonical agricultural label>
```

**Target**

```text
Answer: tomato tomato yellow leaf curl virus
```

### classification.2 `ip102-ip102-v1-1-images-20759-jpg`

- Source: `ip102`
- Verifier: `label`
- Images: `data/raw/ip102/full/ip102_v1.1/images/20759.jpg`

**Prompt**

```text
Identify the insect or pest shown in this agricultural image.

Respond in this format:
Answer: <canonical agricultural label>
```

**Target**

```text
Answer: aphids
```

### classification.3 `ip102-ip102-v1-1-images-72510-jpg`

- Source: `ip102`
- Verifier: `label`
- Images: `data/raw/ip102/full/ip102_v1.1/images/72510.jpg`

**Prompt**

```text
Identify the insect or pest shown in this agricultural image.

Respond in this format:
Answer: <canonical agricultural label>
```

**Target**

```text
Answer: Cicadellidae
```

### classification.4 `ip102-ip102-v1-1-images-15949-jpg`

- Source: `ip102`
- Verifier: `label`
- Images: `data/raw/ip102/full/ip102_v1.1/images/15949.jpg`

**Prompt**

```text
Identify the insect or pest shown in this agricultural image.

Respond in this format:
Answer: <canonical agricultural label>
```

**Target**

```text
Answer: corn borer
```

### classification.5 `ip102-ip102-v1-1-images-52667-jpg`

- Source: `ip102`
- Verifier: `label`
- Images: `data/raw/ip102/full/ip102_v1.1/images/52667.jpg`

**Prompt**

```text
Identify the insect or pest shown in this agricultural image.

Respond in this format:
Answer: <canonical agricultural label>
```

**Target**

```text
Answer: Miridae
```

### classification.6 `plantvillage-train-001156`

- Source: `plantvillage`
- Verifier: `label`
- Images: `data/raw/plantvillage/full/images/train/001156.png`

**Prompt**

```text
Identify the crop issue or pest in this agricultural image.

Respond in this format:
Answer: <canonical agricultural label>
```

**Target**

```text
Answer: apple cedar apple rust
```

### classification.7 `ip102-ip102-v1-1-images-66928-jpg`

- Source: `ip102`
- Verifier: `label`
- Images: `data/raw/ip102/full/ip102_v1.1/images/66928.jpg`

**Prompt**

```text
Identify the insect or pest shown in this agricultural image.

Respond in this format:
Answer: <canonical agricultural label>
```

**Target**

```text
Answer: Lawana imitata Melichar
```

### classification.8 `ip102-ip102-v1-1-images-26209-jpg`

- Source: `ip102`
- Verifier: `label`
- Images: `data/raw/ip102/full/ip102_v1.1/images/26209.jpg`

**Prompt**

```text
Identify the insect or pest shown in this agricultural image.

Respond in this format:
Answer: <canonical agricultural label>
```

**Target**

```text
Answer: flea beetle
```

### classification.9 `ip102-ip102-v1-1-images-70906-jpg`

- Source: `ip102`
- Verifier: `label`
- Images: `data/raw/ip102/full/ip102_v1.1/images/70906.jpg`

**Prompt**

```text
Identify the insect or pest shown in this agricultural image.

Respond in this format:
Answer: <canonical agricultural label>
```

**Target**

```text
Answer: Cicadellidae
```

### classification.10 `ip102-ip102-v1-1-images-24930-jpg`

- Source: `ip102`
- Verifier: `label`
- Images: `data/raw/ip102/full/ip102_v1.1/images/24930.jpg`

**Prompt**

```text
Identify the insect or pest shown in this agricultural image.

Respond in this format:
Answer: <canonical agricultural label>
```

**Target**

```text
Answer: longlegged spider mite
```

### classification.11 `ip102-ip102-v1-1-images-74203-jpg`

- Source: `ip102`
- Verifier: `label`
- Images: `data/raw/ip102/full/ip102_v1.1/images/74203.jpg`

**Prompt**

```text
Identify the insect or pest shown in this agricultural image.

Respond in this format:
Answer: <canonical agricultural label>
```

**Target**

```text
Answer: Cicadellidae
```

### classification.12 `plantvillage-train-043301`

- Source: `plantvillage`
- Verifier: `label`
- Images: `data/raw/plantvillage/full/images/train/043301.png`

**Prompt**

```text
Identify the crop issue or pest in this agricultural image.

Respond in this format:
Answer: <canonical agricultural label>
```

**Target**

```text
Answer: tomato healthy
```

### classification.13 `plantvillage-train-037682`

- Source: `plantvillage`
- Verifier: `label`
- Images: `data/raw/plantvillage/full/images/train/037682.png`

**Prompt**

```text
Identify the crop issue or pest in this agricultural image.

Respond in this format:
Answer: <canonical agricultural label>
```

**Target**

```text
Answer: tomato tomato yellow leaf curl virus
```

### classification.14 `plantvillage-train-003292`

- Source: `plantvillage`
- Verifier: `label`
- Images: `data/raw/plantvillage/full/images/train/003292.png`

**Prompt**

```text
Identify the crop issue or pest in this agricultural image.

Respond in this format:
Answer: <canonical agricultural label>
```

**Target**

```text
Answer: blueberry healthy
```

### classification.15 `plantvillage-train-010240`

- Source: `plantvillage`
- Verifier: `label`
- Images: `data/raw/plantvillage/full/images/train/010240.png`

**Prompt**

```text
Identify the crop issue or pest in this agricultural image.

Respond in this format:
Answer: <canonical agricultural label>
```

**Target**

```text
Answer: grape esca black measles
```

### classification.16 `plantvillage-train-013953`

- Source: `plantvillage`
- Verifier: `label`
- Images: `data/raw/plantvillage/full/images/train/013953.png`

**Prompt**

```text
Identify the crop issue or pest in this agricultural image.

Respond in this format:
Answer: <canonical agricultural label>
```

**Target**

```text
Answer: orange haunglongbing citrus greening
```

### classification.17 `plantvillage-train-009680`

- Source: `plantvillage`
- Verifier: `label`
- Images: `data/raw/plantvillage/full/images/train/009680.png`

**Prompt**

```text
Identify the crop issue or pest in this agricultural image.

Respond in this format:
Answer: <canonical agricultural label>
```

**Target**

```text
Answer: grape esca black measles
```

### classification.18 `ip102-ip102-v1-1-images-68292-jpg`

- Source: `ip102`
- Verifier: `label`
- Images: `data/raw/ip102/full/ip102_v1.1/images/68292.jpg`

**Prompt**

```text
Identify the insect or pest shown in this agricultural image.

Respond in this format:
Answer: <canonical agricultural label>
```

**Target**

```text
Answer: Chlumetia transversa
```

### classification.19 `ip102-ip102-v1-1-images-44629-jpg`

- Source: `ip102`
- Verifier: `label`
- Images: `data/raw/ip102/full/ip102_v1.1/images/44629.jpg`

**Prompt**

```text
Identify the insect or pest shown in this agricultural image.

Respond in this format:
Answer: <canonical agricultural label>
```

**Target**

```text
Answer: Lycorma delicatula
```

### classification.20 `plantvillage-train-007038`

- Source: `plantvillage`
- Verifier: `label`
- Images: `data/raw/plantvillage/full/images/train/007038.png`

**Prompt**

```text
Identify the crop issue or pest in this agricultural image.

Respond in this format:
Answer: <canonical agricultural label>
```

**Target**

```text
Answer: corn maize northern leaf blight
```


## consultation

### consultation.1 `mirage-#778503`

- Source: `mirage`
- Verifier: `exact_match`
- Images: `data/raw/mirage/full/images/MMST_Standard/train/MMST_Standard-train-011323-01.png`

**Prompt**

```text
Please answer the following user question. You should first analyze the provided image, mentioning any visible clues or observations. Then, present the identification result. Write the entire content as one coherent paragraph (analysis + results).
User: I saw what appears to be a yellow fungi growing on kinnikinnick and the ground in my yard today. (picture attached) Is this yellow growth on kinnikinnick something to be concerned about? There is much more kinnikinnick in the yard.

Respond using these line-start section headers exactly once:
Diagnosis:
Evidence:
Uncertainty:
Management:
Follow-up:
```

**Target**

```text
Diagnosis: The image shows bright yellow, irregularly shaped, spongy or crusty masses growing on and around the kinnikinnick plants and the surrounding ground. This type of growth is characteristic of slime mold, which is a nonparasitic organism that thrives on organic debris, often flourishing during moist conditions. The slime mold lives superficially on the plant surfaces and nearby soil but does not invade the plant tissue or cause harm. It appears suddenly and is temporary, eventually drying up and disappearing on its own. There is no need for concern or treatment, as slime mold does not damage kinnikinnick or other plants. If desired, it can be physically removed by disturbing or raking the masses for aesthetic reasons only.
Evidence: Visible symptoms should be checked against the image and crop context.
Uncertainty: Image-only assessment; confirm with field context before treatment.
Management: No source-specific management step was provided; recommend local extension follow-up.
Follow-up: Share close-up images and crop history if symptoms progress or the diagnosis is uncertain.
```

### consultation.2 `agbase-agbase-840024`

- Source: `agbase`
- Verifier: `structured`
- Images: `data/raw/agbase/full/840024/840024_1.jpg`, `data/raw/agbase/full/840024/840024_2.jpg`

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
Diagnosis: hydrangea
Evidence: Visible symptoms should be checked against the image and crop context.
Uncertainty: Image-only assessment; confirm with field context before treatment.
Management: No source-specific management step was provided; recommend local extension follow-up.
Follow-up: Share close-up images and crop history if symptoms progress or the diagnosis is uncertain.
```

### consultation.3 `agbase-agbase-811781`

- Source: `agbase`
- Verifier: `structured`
- Images: `data/raw/agbase/full/811781/811781_1.jpg`

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
Diagnosis: apple
Evidence: apples come off the tree with a slit deep in the flesh
Uncertainty: Image-only assessment; confirm with field context before treatment.
Management: No source-specific management step was provided; recommend local extension follow-up.
Follow-up: Share close-up images and crop history if symptoms progress or the diagnosis is uncertain.
```

### consultation.4 `mirage-#344521`

- Source: `mirage`
- Verifier: `exact_match`
- Images: `data/raw/mirage/full/images/MMST_Standard/train/MMST_Standard-train-005122-01.png`, `data/raw/mirage/full/images/MMST_Standard/train/MMST_Standard-train-005122-02.png`

**Prompt**

```text
Please answer the following user question. You should first analyze the provided image, mentioning any visible clues or observations. Then, present the identification result. Write the entire content as one coherent paragraph (analysis + results).
User: Can't be lemon mint? Grew in the shadiest part of the garden. The app suggested lemon mint, can't be right. I am about to plant it in another location and need to know where: shady/sunny, rich or dry soil etc. Judging by last year it doesn't have flowers.

Respond using these line-start section headers exactly once:
Diagnosis:
Evidence:
Uncertainty:
Management:
Follow-up:
```

**Target**

```text
Diagnosis: The images show a plant with thick, waxy, variegated leaves that are heart-shaped with a distinctive green center and yellowish margins. The leaves are arranged oppositely along a sturdy, slightly fuzzy stem, and no flowers are visible, which matches your observation from last year. These characteristics align with varieties of Sedum, a succulent genus known for such foliage traits. Sedums typically prefer well-drained soil and can tolerate a range of light conditions, from full sun to partial shade, though variegated forms often do better with some protection from intense direct sunlight to prevent leaf scorch. Since your plant thrived in a shady part of the garden, it would be wise to replicate similar conditions—partial shade with moderately rich, well-drained soil—for transplanting. Avoid overly dry soil, as Sedums do appreciate some moisture but dislike waterlogged roots. This plant is unlikely to be lemon mint, which usually has softer, aromatic leaves without variegation and tends to flower more abundantly. Therefore, what you have is best identified as a variegated Sedum, possibly Sedum stoloniferum or a related cultivar, and it should be planted in a spot with light shade and well-draining soil to maintain its health.
Evidence: Visible symptoms should be checked against the image and crop context.
Uncertainty: Image-only assessment; confirm with field context before treatment.
Management: No source-specific management step was provided; recommend local extension follow-up.
Follow-up: Share close-up images and crop history if symptoms progress or the diagnosis is uncertain.
```

### consultation.5 `agbase-agbase-746879`

- Source: `agbase`
- Verifier: `structured`
- Images: `data/raw/agbase/full/746879/746879_2.jpg`, `data/raw/agbase/full/746879/746879_1.jpg`

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
Diagnosis: rose rosette and possibly stem canker
Evidence: excessive thorniness, dense thorns, brown lesions on stems, witches broom (many stems growing out from one main stem)
Uncertainty: Image-only assessment; confirm with field context before treatment.
Management: remove and dispose of the diseased rose in landfill trash
Follow-up: Share close-up images and crop history if symptoms progress or the diagnosis is uncertain.
```

### consultation.6 `agbase-agbase-845891`

- Source: `agbase`
- Verifier: `structured`
- Images: `data/raw/agbase/full/845891/845891_1.jpg`, `data/raw/agbase/full/845891/845891_2.jpg`

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
Diagnosis: sandbar willow
Evidence: Visible symptoms should be checked against the image and crop context.
Uncertainty: Image-only assessment; confirm with field context before treatment.
Management: manual removal in the yard with a weed tool that removes the root and plant. keep mowing them down. minimizing the larger plants along the pond may also help. there is no practical way to completely stop them from appearing without digging up all the underground roots.
Follow-up: Share close-up images and crop history if symptoms progress or the diagnosis is uncertain.
```

### consultation.7 `agbase-agbase-836179`

- Source: `agbase`
- Verifier: `structured`
- Images: `data/raw/agbase/full/836179/836179_1.jpg`, `data/raw/agbase/full/836179/836179_2.jpg`

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
Diagnosis: oregano, hyssop, bee balm
Evidence: Visible symptoms should be checked against the image and crop context.
Uncertainty: Image-only assessment; confirm with field context before treatment.
Management: No source-specific management step was provided; recommend local extension follow-up.
Follow-up: Share close-up images and crop history if symptoms progress or the diagnosis is uncertain.
```

### consultation.8 `agbase-agbase-585078`

- Source: `agbase`
- Verifier: `structured`
- Images: `data/raw/agbase/full/585078/585078_3.jpg`, `data/raw/agbase/full/585078/585078_2.jpg`, `data/raw/agbase/full/585078/585078_1.jpg`

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
Diagnosis: canker stain
Evidence: brown leaves
Uncertainty: Image-only assessment; confirm with field context before treatment.
Management: prune off the branch with the brown leaves to look for staining, if canker is suspected, remove the tree and dispose of it by burning to avoid spreading the fungus to other trees
Follow-up: Share close-up images and crop history if symptoms progress or the diagnosis is uncertain.
```

### consultation.9 `mirage-#708270`

- Source: `mirage`
- Verifier: `exact_match`
- Images: `data/raw/mirage/full/images/MMST_Standard/train/MMST_Standard-train-002532-01.png`, `data/raw/mirage/full/images/MMST_Standard/train/MMST_Standard-train-002532-02.png`

**Prompt**

```text
We have some tree root fungus growing under our conifers in our yard and aren't sure whether we should be concerned for the trees. Attached are pics of the fungus and one of the trees (since I don't know exactly what kind of trees they are). The trees have had the new needles on the ends of their branches die off some, but are otherwise fine.

Respond using these line-start section headers exactly once:
Diagnosis:
Evidence:
Uncertainty:
Management:
Follow-up:
```

**Target**

```text
Diagnosis: The tree looks like it is an Atlas or Deodor Cedar. It looks very healthy and the fungus growing in the soil is not impacting its health. If you see fungus growing on the trunk, you'd have reason for concern.
Evidence: Visible symptoms should be checked against the image and crop context.
Uncertainty: Image-only assessment; confirm with field context before treatment.
Management: No source-specific management step was provided; recommend local extension follow-up.
Follow-up: Share close-up images and crop history if symptoms progress or the diagnosis is uncertain.
```

### consultation.10 `agbase-agbase-811686`

- Source: `agbase`
- Verifier: `structured`
- Images: `data/raw/agbase/full/811686/811686_1.jpg`

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
Diagnosis: root rots
Evidence: big sections looked like they were drying out
Uncertainty: Image-only assessment; confirm with field context before treatment.
Management: choose a different shrub as a replacement. improving drainage is advised.
Follow-up: Share close-up images and crop history if symptoms progress or the diagnosis is uncertain.
```

### consultation.11 `agbase-agbase-809906`

- Source: `agbase`
- Verifier: `structured`
- Images: `data/raw/agbase/full/809906/809906_1.jpg`, `data/raw/agbase/full/809906/809906_2.jpg`

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
Diagnosis: none
Evidence: Visible symptoms should be checked against the image and crop context.
Uncertainty: Image-only assessment; confirm with field context before treatment.
Management: submit an insect sample to the uconn plant diagnostic lab for identification
Follow-up: Share close-up images and crop history if symptoms progress or the diagnosis is uncertain.
```

### consultation.12 `mirage-#408068`

- Source: `mirage`
- Verifier: `exact_match`
- Images: `data/raw/mirage/full/images/MMST_Standard/train/MMST_Standard-train-011556-01.png`, `data/raw/mirage/full/images/MMST_Standard/train/MMST_Standard-train-011556-02.png`

**Prompt**

```text
Please answer the following user question. You should first analyze the provided image, mentioning any visible clues or observations. Then, present the identification result. Write the entire content as one coherent paragraph (analysis + results).
User: My hostas have started turning brown on the outer edges of some leaves. What could be causing the browning of my hostas?

Respond using these line-start section headers exactly once:
Diagnosis:
Evidence:
Uncertainty:
Management:
Follow-up:
```

**Target**

```text
Diagnosis: The images show hosta leaves with browning and dry areas primarily around the outer edges of some leaves. The affected leaf margins have irregular, tan to brown patches that are dry and slightly curled. This symptom pattern aligns with either sunscald—damage caused by prolonged exposure to intense sunlight leading to leaf tissue drying and browning—or the early stages of anthracnose, a fungal disease (Sphaceloma de Bary, 1874) that can cause similar marginal browning and irregular dead patches. Given the visible damage limited to leaf edges without extensive necrotic spots or lesions spreading inward, sunscald is a likely cause, especially if plants are exposed to strong direct sun. To manage this, consider relocating your hostas to a shadier area or providing shade during peak sunlight hours. If you suspect anthracnose, remove and dispose of affected leaves, ensure good air circulation, avoid overhead watering, and monitor for progression of symptoms to decide if fungicidal treatment is necessary.
Evidence: Visible symptoms should be checked against the image and crop context.
Uncertainty: Image-only assessment; confirm with field context before treatment.
Management: No source-specific management step was provided; recommend local extension follow-up.
Follow-up: Share close-up images and crop history if symptoms progress or the diagnosis is uncertain.
```

### consultation.13 `agbase-agbase-682011`

- Source: `agbase`
- Verifier: `structured`
- Images: `data/raw/agbase/full/682011/682011_1.jpg`, `data/raw/agbase/full/682011/682011_3.jpg`, `data/raw/agbase/full/682011/682011_2.jpg`

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
Diagnosis: frost damage, leaf spot fungus
Evidence: dark and dropping leaves, brown leaves, yellow leaves
Uncertainty: Image-only assessment; confirm with field context before treatment.
Management: wait until new growth appears before pruning, do not fertilize, lightly water, shape in april, leave a few inches on liriope, be patient
Follow-up: Share close-up images and crop history if symptoms progress or the diagnosis is uncertain.
```

### consultation.14 `mirage-#576427`

- Source: `mirage`
- Verifier: `exact_match`
- Images: `data/raw/mirage/full/images/MMST_Standard/train/MMST_Standard-train-003864-01.png`, `data/raw/mirage/full/images/MMST_Standard/train/MMST_Standard-train-003864-02.png`

**Prompt**

```text
Please answer the following user question. You should first analyze the provided image, mentioning any visible clues or observations. Then, present the identification result. Write the entire content as one coherent paragraph (analysis + results).
User: My garden has an issue with hot peppers. The leaves are shriveled and twisted looking. What gives?

Respond using these line-start section headers exactly once:
Diagnosis:
Evidence:
Uncertainty:
Management:
Follow-up:
```

**Target**

```text
Diagnosis: The photos show hot pepper leaves that are noticeably curled, twisted, and somewhat shriveled, with some leaf tips exhibiting browning and dried edges consistent with sunburn stress. This curling and distortion pattern strongly suggests an infestation by sucking insects such as aphids, whiteflies, or mites, which feed on the plant’s sap and cause leaf deformation. These pests are often found on the undersides of leaves or within the curled foliage, and their presence can exacerbate stress symptoms like sunburn damage. To manage this issue, regularly inspect the plants for tiny insects, spray with insecticidal soap to control the pests, and prune off heavily damaged leaves. Additionally, applying a balanced liquid fertilizer like fish emulsion will help boost plant vigor, which reduces susceptibility. Maintaining consistent moisture and applying mulch can prevent further stress, while shade cloth can protect the plants from excessive sun exposure. This approach should help relieve the combined pressures of insect feeding and environmental stress affecting your hot peppers.
Evidence: Visible symptoms should be checked against the image and crop context.
Uncertainty: Image-only assessment; confirm with field context before treatment.
Management: No source-specific management step was provided; recommend local extension follow-up.
Follow-up: Share close-up images and crop history if symptoms progress or the diagnosis is uncertain.
```

### consultation.15 `agbase-agbase-807634`

- Source: `agbase`
- Verifier: `structured`
- Images: `data/raw/agbase/full/807634/807634_1.jpg`, `data/raw/agbase/full/807634/807634_2.jpg`

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
Diagnosis: spruce
Evidence: Visible symptoms should be checked against the image and crop context.
Uncertainty: Image-only assessment; confirm with field context before treatment.
Management: check the conditions the spruce was kept in before, keep the soil from getting too wet or too dry, keep the needles dry to prevent fungal spores from spreading. use a magnifying lens to check for fungal disease by comparing the stomata on healthy and unhealthy needles. if fungal bodies are present, apply a registered fungicide per label instructions.
Follow-up: Share close-up images and crop history if symptoms progress or the diagnosis is uncertain.
```

### consultation.16 `agbase-agbase-752119`

- Source: `agbase`
- Verifier: `structured`
- Images: `data/raw/agbase/full/752119/752119_2.jpg`, `data/raw/agbase/full/752119/752119_1.jpg`, `data/raw/agbase/full/752119/752119_3.jpg`

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
Evidence: one main upper limb did not leaf this spring, bark on the dead branches looks odd
Uncertainty: Image-only assessment; confirm with field context before treatment.
Management: contact a licensed arborist for inspection and recommendations
Follow-up: Share close-up images and crop history if symptoms progress or the diagnosis is uncertain.
```

### consultation.17 `agbase-agbase-638667`

- Source: `agbase`
- Verifier: `structured`
- Images: `data/raw/agbase/full/638667/638667_3.jpg`, `data/raw/agbase/full/638667/638667_2.jpg`, `data/raw/agbase/full/638667/638667_1.jpg`

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
Diagnosis: spruce trees
Evidence: thinning
Uncertainty: Image-only assessment; confirm with field context before treatment.
Management: hire a certified arborist for a complete diagnosis and a plant care plan.
Follow-up: Share close-up images and crop history if symptoms progress or the diagnosis is uncertain.
```

### consultation.18 `agbase-agbase-642712`

- Source: `agbase`
- Verifier: `structured`
- Images: `data/raw/agbase/full/642712/642712_2.jpg`, `data/raw/agbase/full/642712/642712_1.jpg`, `data/raw/agbase/full/642712/642712_3.jpg`

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
Diagnosis: anthracnose
Evidence: widespread leaf damage
Uncertainty: Image-only assessment; confirm with field context before treatment.
Management: No source-specific management step was provided; recommend local extension follow-up.
Follow-up: Share close-up images and crop history if symptoms progress or the diagnosis is uncertain.
```

### consultation.19 `agbase-agbase-750442`

- Source: `agbase`
- Verifier: `structured`
- Images: `data/raw/agbase/full/750442/750442_1.jpg`

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
Diagnosis: white avens
Evidence: Visible symptoms should be checked against the image and crop context.
Uncertainty: Image-only assessment; confirm with field context before treatment.
Management: No source-specific management step was provided; recommend local extension follow-up.
Follow-up: Share close-up images and crop history if symptoms progress or the diagnosis is uncertain.
```

### consultation.20 `mirage-#744130`

- Source: `mirage`
- Verifier: `exact_match`
- Images: `data/raw/mirage/full/images/MMST_Standard/train/MMST_Standard-train-009310-01.png`, `data/raw/mirage/full/images/MMST_Standard/train/MMST_Standard-train-009310-02.png`

**Prompt**

```text
Please answer the following user question. You should first analyze the provided image, mentioning any visible clues or observations. Then, present the identification result. Write the entire content as one coherent paragraph (analysis + results).
User: Can you help me identify what bugs are causing a small bug infestation on my indoor lavender plant? I'm wondering if it is safe to put this container plant outside (or if the bugs may infect other plants)? I've cut most of the branches down and put into the trash (so the bugs I could see are gone).

Respond using these line-start section headers exactly once:
Diagnosis:
Evidence:
Uncertainty:
Management:
Follow-up:
```

**Target**

```text
Diagnosis: The images show a lavender plant with numerous fine, silvery web-like threads stretched between the stems and leaves, along with numerous tiny reddish-brown specks that appear to be moving. The plant’s leaves look curled, desiccated, and unhealthy, all classic signs of a spider mite infestation. These microscopic arachnids spin webs as they feed on plant sap, causing significant stress and damage to the plant. The infestation visible in the photos matches the tumid spider mite (Tetranychus tumidus), a common pest on indoor and container plants like lavender. Because spider mites multiply rapidly and can easily spread to nearby plants, it’s recommended to thoroughly treat the plant with insecticidal soap, especially targeting leaf undersides, and keep humidity higher to discourage mite growth. Before moving the lavender outdoors, isolate it and monitor for remaining mites to prevent spreading the infestation to other plants. Your steps removing infested branches are good, but ongoing treatment and caution are needed to fully resolve the problem and protect your garden.
Evidence: Visible symptoms should be checked against the image and crop context.
Uncertainty: Image-only assessment; confirm with field context before treatment.
Management: No source-specific management step was provided; recommend local extension follow-up.
Follow-up: Share close-up images and crop history if symptoms progress or the diagnosis is uncertain.
```


## vqa

### vqa.1 `plantvillage_vqa-image_001382.JPG-004026`

- Source: `plantvillage_vqa`
- Verifier: `exact_match`
- Images: `data/raw/plantvillage_vqa/full/images/train/image_001382.JPG`

**Prompt**

```text
What rationale explains these symptoms?

Respond in this format:
Answer: <short answer>
```

**Target**

```text
Answer: Diagnosis: Huanglongbing (HLB), or citrus greening.
```

### vqa.2 `plantvillage_vqa-image_041756.JPG-122757`

- Source: `plantvillage_vqa`
- Verifier: `exact_match`
- Images: `data/raw/plantvillage_vqa/full/images/train/image_041756.JPG`

**Prompt**

```text
What disease, if any, is visible on this Soybean leaf?

Respond in this format:
Answer: <short answer>
```

**Target**

```text
Answer: This is a healthy plant leaf.
```

### vqa.3 `plantvillage_vqa-image_025806.JPG-075712`

- Source: `plantvillage_vqa`
- Verifier: `exact_match`
- Images: `data/raw/plantvillage_vqa/full/images/train/image_025806.JPG`

**Prompt**

```text
What specific factors have led to this tomato leaf's unhealthy manifestation?

Respond in this format:
Answer: <short answer>
```

**Target**

```text
Answer: These 'bullseye' lesions are a tell-tale sign of Early Blight.
```

### vqa.4 `plantvillage_vqa-image_035077.JPG-103189`

- Source: `plantvillage_vqa`
- Verifier: `exact_match`
- Images: `data/raw/plantvillage_vqa/full/images/train/image_035077.JPG`

**Prompt**

```text
Would you deliver a comprehensive visual analysis of this plant specimen?

Respond in this format:
Answer: <short answer>
```

**Target**

```text
Answer: This tomato leaf shows classic signs of TYLCV, like yellowing and curling.
```

### vqa.5 `plantvillage_vqa-image_037322.JPG-109768`

- Source: `plantvillage_vqa`
- Verifier: `exact_match`
- Images: `data/raw/plantvillage_vqa/full/images/train/image_037322.JPG`

**Prompt**

```text
Does this foliage qualify as infected tissue?

Respond in this format:
Answer: <Yes or No>
```

**Target**

```text
Answer: Yes
```

### vqa.6 `plantvillage_vqa-image_041766.JPG-122792`

- Source: `plantvillage_vqa`
- Verifier: `exact_match`
- Images: `data/raw/plantvillage_vqa/full/images/train/image_041766.JPG`

**Prompt**

```text
Can you identify the origin of these visible defects?

Respond in this format:
Answer: <short answer>
```

**Target**

```text
Answer: This is citrus greening disease.
```

### vqa.7 `plantvillage_vqa-image_034318.JPG-100950`

- Source: `plantvillage_vqa`
- Verifier: `exact_match`
- Images: `data/raw/plantvillage_vqa/full/images/train/image_034318.JPG`

**Prompt**

```text
What observable attributes and identifying characteristics are evident in this plant specimen?

Respond in this format:
Answer: <short answer>
```

**Target**

```text
Answer: The symptoms, especially the olive-green mold on the leaf underside, indicate Leaf Mold.
```

### vqa.8 `plantvillage_vqa-image_000943.JPG-002633`

- Source: `plantvillage_vqa`
- Verifier: `exact_match`
- Images: `data/raw/plantvillage_vqa/full/images/train/image_000943.JPG`

**Prompt**

```text
Communicate all available information concerning this plant's existing state.

Respond in this format:
Answer: <short answer>
```

**Target**

```text
Answer: Diagnosis: Huanglongbing (HLB), or citrus greening.
```

### vqa.9 `plantvillage_vqa-image_026750.JPG-179834`

- Source: `plantvillage_vqa`
- Verifier: `exact_match`
- Images: `data/raw/plantvillage_vqa/full/images/train/image_026750.JPG`

**Prompt**

```text
Are there small, dark, saturated lesions with yellow borders visible?

Respond in this format:
Answer: <Yes or No>
```

**Target**

```text
Answer: No
```

### vqa.10 `plantvillage_vqa-image_007082.JPG-172332`

- Source: `plantvillage_vqa`
- Verifier: `exact_match`
- Images: `data/raw/plantvillage_vqa/full/images/train/image_007082.JPG`

**Prompt**

```text
Can you observe tan spots surrounded by dark border zones?

Respond in this format:
Answer: <Yes or No>
```

**Target**

```text
Answer: No
```

### vqa.11 `plantvillage_vqa-image_043956.JPG-191021`

- Source: `plantvillage_vqa`
- Verifier: `exact_match`
- Images: `data/raw/plantvillage_vqa/full/images/train/image_043956.JPG`

**Prompt**

```text
Are there small round lesions with dark edges and light centers?

Respond in this format:
Answer: <Yes or No>
```

**Target**

```text
Answer: Yes
```

### vqa.12 `plantvillage_vqa-image_003929.JPG-011397`

- Source: `plantvillage_vqa`
- Verifier: `exact_match`
- Images: `data/raw/plantvillage_vqa/full/images/train/image_003929.JPG`

**Prompt**

```text
Compare this plant to healthy norms and identify the distinguishing factors.

Respond in this format:
Answer: <short answer>
```

**Target**

```text
Answer: A healthy version of this plant would not exhibit the evidence of orange or brown pustular structures.
```

### vqa.13 `plantvillage_vqa-image_016668.JPG-048810`

- Source: `plantvillage_vqa`
- Verifier: `exact_match`
- Images: `data/raw/plantvillage_vqa/full/images/train/image_016668.JPG`

**Prompt**

```text
Can you diagnose disease presence in this leaf?

Respond in this format:
Answer: <Yes or No>
```

**Target**

```text
Answer: Yes
```

### vqa.14 `plantvillage_vqa-image_020286.JPG-059301`

- Source: `plantvillage_vqa`
- Verifier: `exact_match`
- Images: `data/raw/plantvillage_vqa/full/images/train/image_020286.JPG`

**Prompt**

```text
What pathological condition do these visual signs indicate?

Respond in this format:
Answer: <short answer>
```

**Target**

```text
Answer: The stippling and fine webbing are clear signs of a Spider Mite infestation.
```

### vqa.15 `plantvillage_vqa-image_033193.JPG-097708`

- Source: `plantvillage_vqa`
- Verifier: `exact_match`
- Images: `data/raw/plantvillage_vqa/full/images/train/image_033193.JPG`

**Prompt**

```text
Would you provide an exact disease diagnosis based on this image?

Respond in this format:
Answer: <short answer>
```

**Target**

```text
Answer: The plant is suffering from a fungal infection: Septoria Leaf Spot.
```

### vqa.16 `plantvillage_vqa-image_029804.JPG-087691`

- Source: `plantvillage_vqa`
- Verifier: `exact_match`
- Images: `data/raw/plantvillage_vqa/full/images/train/image_029804.JPG`

**Prompt**

```text
What factors are contributing to the deteriorated condition of this tomato foliage?

Respond in this format:
Answer: <short answer>
```

**Target**

```text
Answer: This is damage from Spider Mites.
```

### vqa.17 `plantvillage_vqa-image_013154.JPG-038488`

- Source: `plantvillage_vqa`
- Verifier: `exact_match`
- Images: `data/raw/plantvillage_vqa/full/images/train/image_013154.JPG`

**Prompt**

```text
Can you observe disease-caused abnormalities?

Respond in this format:
Answer: <Yes or No>
```

**Target**

```text
Answer: No
```

### vqa.18 `plantvillage_vqa-image_029787.JPG-087625`

- Source: `plantvillage_vqa`
- Verifier: `exact_match`
- Images: `data/raw/plantvillage_vqa/full/images/train/image_029787.JPG`

**Prompt**

```text
Categorize this plant according to its scientific nomenclature.

Respond in this format:
Answer: <short answer>
```

**Target**

```text
Answer: Tomato
```

### vqa.19 `plantvillage_vqa-image_014352.JPG-041983`

- Source: `plantvillage_vqa`
- Verifier: `exact_match`
- Images: `data/raw/plantvillage_vqa/full/images/train/image_014352.JPG`

**Prompt**

```text
Are there visible signs of microbial activity present?

Respond in this format:
Answer: <Yes or No>
```

**Target**

```text
Answer: Yes
```

### vqa.20 `plantvillage_vqa-image_018007.JPG-052675`

- Source: `plantvillage_vqa`
- Verifier: `exact_match`
- Images: `data/raw/plantvillage_vqa/full/images/train/image_018007.JPG`

**Prompt**

```text
Which causal agents are responsible for this tomato leaf's abnormal appearance?

Respond in this format:
Answer: <short answer>
```

**Target**

```text
Answer: The leaf shows symptoms of Early Blight.
```

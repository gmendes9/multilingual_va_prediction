# Quantifying Valence and Arousal in Text with Multilingual Pre-trained Transformers 
Repository for Quantifying Valence and Arousal in Text with Multilingual Pre-trained Transformers 


## Dataset
The dataset proposed in this paper was built collecting 34 different public datasets of annotated data for the emotional dimensions of Valence and Arousal.
All the datasets used free to use for research purposes, although some require an authorization to use, and/or individual acceptance of the respective terms of use. For this reason, we cannot publicly provide the dataset we used to train our models. 

As detailed in the Paper, our dataset is a .csv file with three columns, namely "text", "valence", and "arousal".
To reproduce our dataset, follow this procedure:
- Retrieve the 34 original datasets from the Dataset Sorces below. The datasets come in various different file formats, such as .csv, .xlsx, .txt, etc.
- Filter the relevant data:
  - **text**: Word or short text content
  - **valence** and **arousal**: We simply used the Valence and Arousal Mean values.
- Normalize the Valence and Arousal scores between **zero** and **one**, using the following formula.
  - $z_i = (x_i - \textrm{min}(x)) / (\textrm{max}(x) - \textrm{min}(x))$
  - $z_i$ denotes the normalized value, $x_i$ the original value, and $\textrm{min}$ and $\textrm{max}$ denote the extremes of the scales in which the original scores were rated on.


### Dataset Sources
#### EmoBank
- **Source:** EmoBank: Studying the Impact of Annotation Perspective and Representation Format on Dimensional Emotion Analysis
  - https://aclanthology.org/E17-2092/
- **Repository:** https://github.com/JULIELab/EmoBank
- **Download directly here:** https://github.com/JULIELab/EmoBank/raw/master/corpus/emobank.csv

#### IEMOCAP
- **Source:** IEMOCAP: Interactive emotional dyadic motion capture database
  - https://sail.usc.edu/iemocap/Busso_2008_iemocap.pdf
- **Repository:** https://sail.usc.edu/iemocap/iemocap_release.htm
  - To obtain the IEMOCAP data you need to fill out an electronic release form.
  
#### Facebook Posts
- **Source:** Modelling Valence and Arousal in Facebook posts
  - https://aclanthology.org/W16-0404/https://aclanthology.org/W16-0404/
- **Repository:** https://github.com/wwbp/additional_data_sets/tree/master/valence_arousal
- **Download directly here:** https://github.com/wwbp/additional_data_sets/raw/master/valence_arousal/dataset-fb-valence-arousal-anon.csv

#### EmoTales
- **Source:** EmoTales: creating a corpus of folk tales with emotional annotations
  - https://link.springer.com/article/10.1007/s10579-011-9140-5
- **Repository:** Request the dataset by contacting the author, Virgina Francisco, at virginia@fdi.ucm.es

#### ANET
- **Source:** Affective Norms for English Text (ANET)
  - https://csea.phhp.ufl.edu/media/anetmessage.html
- **Repository:** https://csea.phhp.ufl.edu/media/anetmessage.html
  - To obtain the ANET data you need to fill out an electronic release form.

#### PANIG
- **Source:** When emotions are expressed figuratively: Psycholinguistic and Affective Norms of 619 Idioms for German (PANIG)
  - https://link.springer.com/article/10.3758/s13428-015-0581-4
- **Repository:** Data available in the Electronic Supplementary Material (ESM) section of the Springer web page above.
- **Download directly here:** https://static-content.springer.com/esm/art%3A10.3758%2Fs13428-015-0581-4/MediaObjects/13428_2015_581_MOESM1_ESM.xls

#### COMETA sentences
- **Source:** Affective and psycholinguistic norms for German conceptual metaphors (COMETA)
  - https://link.springer.com/article/10.3758/s13428-019-01300-7
- **Repository:** Data available in the Electronic Supplementary Material (ESM) section of the Springer web page above. See ESM 2.
- **Download directly here:** https://static-content.springer.com/esm/art%3A10.3758%2Fs13428-019-01300-7/MediaObjects/13428_2019_1300_MOESM2_ESM.xlsx

#### COMETA stories
- **Source:** Affective and psycholinguistic norms for German conceptual metaphors (COMETA)
  - https://link.springer.com/article/10.3758/s13428-019-01300-7
- **Repository:** Data available in the Electronic Supplementary Material (ESM) section of the Springer web page above. See ESM 2.
- **Download directly here:** https://static-content.springer.com/esm/art%3A10.3758%2Fs13428-019-01300-7/MediaObjects/13428_2019_1300_MOESM2_ESM.xlsx

#### CVAT
- **Source:** Building Chinese Affective Resources in Valence-Arousal Dimensions
  - https://aclanthology.org/N16-1066/
- **Repository:** http://nlp.innobic.yzu.edu.tw/resources/ChineseEmoBank.html
- **Download directly here:** http://nlp.innobic.yzu.edu.tw/resources/chinese-emobank_download.html

#### CVAI
- **Source:** A Dimensional Valence-Arousal-Irony Dataset for Chinese Sentence and Context
  - https://aclanthology.org/2022.rocling-1.19/
- **Repository:** http://nlp.innobic.yzu.edu.tw/resources/chinese-vai_download.html

#### ANPST
- **Source:** Affective Norms for 718 Polish Short Texts (ANPST): Dataset with Affective Ratings for Valence, Arousal, Dominance, Origin, Subjective Significance and Source Dimensions
  - https://www.frontiersin.org/articles/10.3389/fpsyg.2016.01030/full
- **Repository:** https://figshare.com/s/e4b4e339138f07c63153
- **Download directly here:** https://figshare.com/ndownloader/files/5343997?private_link=e4b4e339138f07c63153

#### MAS
- **Source:** Minho Affective Sentences (MAS): Probing the roles of sex, mood, and empathy in affective ratings of verbal stimuli
  - https://link.springer.com/article/10.3758/s13428-016-0726-0
- **Repository:** Data available in the Electronic Supplementary Material (ESM) section of the Springer web page above. See ESM 2.
- **Download directly here:** https://static-content.springer.com/esm/art%3A10.3758%2Fs13428-016-0726-0/MediaObjects/13428_2016_726_MOESM2_ESM.docx

#### Yee
- **Source:** Valence, arousal, familiarity, concreteness, and imageability ratings for 292 two-character Chinese nouns in Cantonese speakers in Hong Kong
  - https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0174569#sec014
- **Repository:** https://figshare.com/articles/dataset/Valence_arousal_familiarity_concreteness_and_imageability_ratings_for_292_two-character_Chinese_nouns_in_Cantonese_speakers_in_Hong_Kong/4791586?file=7883134
- **Download directly here:** https://figshare.com/ndownloader/articles/4791586/versions/1

#### Ćoso et al.
- **Source:** Affective and concreteness norms for 3,022 Croatian words
  - https://journals.sagepub.com/doi/full/10.1177/1747021819834226
- **Repository:** https://www.ucace.com/links/
- **Download directly here:** https://www.ucace.com/app/download/30931812/Supplementary+material_%C4%86oso+et+al.xlsx
 
#### Moors et al.
- **Source:** Norms of valence, arousal, dominance, and age of acquisition for 4,300 Dutch words 
  - https://link.springer.com/article/10.3758/s13428-012-0243-8
- **Repository:** Data available in the Electronic Supplementary Material (ESM) section of the Springer web page above.
- **Download directly here:** https://static-content.springer.com/esm/art%3A10.3758%2Fs13428-012-0243-8/MediaObjects/13428_2012_243_MOESM1_ESM.xlsx

#### Verheyen et al.
- **Source:** Lexicosemantic, affective, and distributional norms for 1,000 Dutch adjectives
  - https://link.springer.com/article/10.3758/s13428-019-01303-4
- **Repository:** https://osf.io/nyg8v/
- **Download directly here:** https://osf.io/download/6zxej/

#### NRC-VAD
- **Source:** Obtaining Reliable Human Ratings of Valence, Arousal, and Dominance for 20,000 English Words
  - https://aclanthology.org/P18-1017/
- **Repository:** http://saifmohammad.com/WebPages/nrc-vad.html
- **Download directly here:** http://saifmohammad.com/WebDocs/Lexicons/NRC-VAD-Lexicon.zip

#### Warriner et al.
- **Source:** Norms of valence, arousal, and dominance for 13,915 English lemmas
  - https://link.springer.com/article/10.3758/s13428-012-0314-x
- **Repository:** Data available in the Electronic Supplementary Material (ESM) section of the Springer web page above.
- **Download directly here:** https://static-content.springer.com/esm/art%3A10.3758%2Fs13428-012-0314-x/MediaObjects/13428_2012_314_MOESM1_ESM.zip

#### Scott et al.
- **Source:** The Glasgow Norms: Ratings of 5,500 words on nine scales
  - https://static-content.springer.com/esm/art%3A10.3758%2Fs13428-012-0314-x/MediaObjects/13428_2012_314_MOESM1_ESM.zip
- **Repository:** Data available in the Electronic Supplementary Material (ESM) section of the Springer web page above.
- **Download directly here:** https://static-content.springer.com/esm/art%3A10.3758%2Fs13428-018-1099-3/MediaObjects/13428_2018_1099_MOESM2_ESM.csv

#### Söderholm et al.
- **Source:** Valence and arousal ratings for 420 Finnish nouns by age and gender
  - https://pubmed.ncbi.nlm.nih.gov/24023650/
- **Repository:** https://figshare.com/articles/dataset/_Valence_and_Arousal_Ratings_for_420_Finnish_Nouns_by_Age_and_Gender_/785492
- **Download directly here:** https://figshare.com/ndownloader/files/1186672

#### Eilola et al.
- **Source:** Affective norms for 210 British English and Finnish nouns
  - https://link.springer.com/article/10.3758/BRM.42.1.134#SecESM1
- **Repository:** Data available in the Electronic Supplementary Material (ESM) section of the Springer web page above.
- **Download directly here:** https://static-content.springer.com/esm/art%3A10.3758%2FBRM.42.1.134/MediaObjects/Eilola-BRM-2010.zip

#### FAN
- **Source:** Affective norms for french words (FAN)
  - https://link.springer.com/article/10.3758/s13428-013-0431-1
- **Repository:** Data available in the Electronic Supplementary Material (ESM) section of the Springer web page above.
- **Download directly here:** https://static-content.springer.com/esm/art%3A10.3758%2Fs13428-013-0431-1/MediaObjects/13428_2013_431_MOESM2_ESM.xlsx

#### FEEL
- **Source:** Valence, arousal, and imagery ratings for 835 French attributes by young, middle-aged, and older adults: The French Emotional Evaluation List (FEEL)
  - https://www.sciencedirect.com/science/article/abs/pii/S1162908812000278
- **Repository:** https://osf.io/u52dy/
- **Download directly here:** https://osf.io/download/ps7te/

#### BAWL-R
- **Source:** The Berlin Affective Word List Reloaded (BAWL-R)
  - https://link.springer.com/article/10.3758/BRM.41.2.534
- **Repository:** https://osf.io/hx6r8/
- **Download directly here:** https://osf.io/download/cspef/

#### ANGST
- **Source:** ANGST: Affective norms for German sentiment terms, derived from the affective norms for English words
  - https://link.springer.com/article/10.3758/s13428-013-0426-y
- **Repository:** Data available in the Electronic Supplementary Material (ESM) section of the Springer web page above.
- **Download directly here:** https://static-content.springer.com/esm/art%3A10.3758%2Fs13428-013-0426-y/MediaObjects/13428_2013_426_MOESM1_ESM.xlsx

#### LANG
- **Source:** Leipzig Affective Norms for German: A reliability study
  - https://link.springer.com/article/10.3758/BRM.42.4.987#SecESM1
- **Repository:** Data available in the Electronic Supplementary Material (ESM) section of the Springer web page above.
- **Download directly here:** https://static-content.springer.com/esm/art%3A10.3758%2FBRM.42.4.987/MediaObjects/Kanske-BRM-2010.zip

#### Italian ANEW
- **Source:** Affective Norms for Italian Words in Older Adults: Age Differences in Ratings of Valence, Arousal and Dominance
  - https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0169472#sec015
- **Repository:** https://figshare.com/articles/dataset/Affective_Norms_for_Italian_Words_in_Older_Adults_Age_Differences_in_Ratings_of_Valence_Arousal_and_Dominance/4512950
- **Download directly here:** https://figshare.com/ndownloader/files/7305791

#### Xu et al.
- **Source:** Valence and arousal ratings for 11,310 simplified Chinese words
  - https://link.springer.com/article/10.3758/s13428-021-01607-4
- **Repository:** Data available in the Electronic Supplementary Material (ESM) section of the Springer web page above.
- **Download directly here:** https://static-content.springer.com/esm/art%3A10.3758%2Fs13428-021-01607-4/MediaObjects/13428_2021_1607_MOESM1_ESM.csv

#### CVAW
- **Source:** Building Chinese Affective Resources in Valence-Arousal Dimensions
  - https://aclanthology.org/N16-1066/
- **Repository:** http://nlp.innobic.yzu.edu.tw/resources/ChineseEmoBank.html
- **Download directly here:** http://nlp.innobic.yzu.edu.tw/resources/chinese-emobank_download.html

#### ANPW_R
- **Source:** Affective Norms for 4900 Polish Words Reload (ANPW_R): Assessments for Valence, Arousal, Dominance, Origin, Significance, Concreteness, Imageability and, Age of Acquisition
  - https://www.frontiersin.org/articles/10.3389/fpsyg.2016.01081/full#h10
- **Repository:** https://figshare.com/articles/dataset/DataSheet1_Affective_Norms_for_4900_Polish_Words_Reload_ANPW_R_Assessments_for_Valence_Arousal_Dominance_Origin_Significance_Concreteness_Imageability_and_Age_of_Acquisition_XLSX/16420035?backTo=/collections/Affective_Norms_for_4900_Polish_Words_Reload_ANPW_R_Assessments_for_Valence_Arousal_Dominance_Origin_Significance_Concreteness_Imageability_and_Age_of_Acquisition/5579574
- **Download directly here:** https://figshare.com/ndownloader/files/30426825

#### NAWL
- **Source:** Nencki Affective Word List (NAWL): the cultural adaptation of the Berlin Affective Word List–Reloaded (BAWL-R) for Polish
  - https://link.springer.com/article/10.3758/s13428-014-0552-1#Sec18
- **Repository:** Data available in the Electronic Supplementary Material (ESM) section of the Springer web page above.
- **Download directly here:** https://static-content.springer.com/esm/art%3A10.3758%2Fs13428-014-0552-1/MediaObjects/13428_2014_552_MOESM1_ESM.xlsx

#### Portuguese ANEW
- **Source:** The adaptation of the Affective Norms for English Words (ANEW) for European Portuguese
  - https://link.springer.com/article/10.3758/s13428-011-0131-7
- **Repository:** Data available in the Electronic Supplementary Material (ESM) section of the Springer web page above.
- **Download directly here:** https://static-content.springer.com/esm/art%3A10.3758%2Fs13428-011-0131-7/MediaObjects/13428_2011_131_MOESM1_ESM.xls

#### Stadthagen-Gonzalez et al.
- **Source:** Norms of valence and arousal for 14,031 Spanish words
  - https://link.springer.com/article/10.3758/s13428-015-0700-2#Sec16
- **Repository:** Data available in the Electronic Supplementary Material (ESM) section of the Springer web page above.
- **Download directly here:** https://static-content.springer.com/esm/art%3A10.3758%2Fs13428-015-0700-2/MediaObjects/13428_2015_700_MOESM1_ESM.csv

#### Kapucu et al.
- **Source:** Turkish Emotional Word Norms for Arousal, Valence, and Discrete Emotion Categories
  - https://journals.sagepub.com/doi/10.1177/0033294118814722?url_ver=Z39.88-2003&rfr_id=ori:rid:crossref.org&rfr_dat=cr_pub%20%200pubmed
- **Repository:** https://osf.io/86a4g/
- **Download directly here:** https://osf.io/download/x5sm8/








## Models
We make available the models we trained, in three different sizes. These three multilingual models support 100 languages.
- DistilBERT
  - 134M parameters.
- XLM-RoBERTa-base
  - 270M parameters.
- XLM-RoBERTa-large
  - 550M parameters.

#### DistilBERT
https://drive.google.com/drive/folders/1a3ToFHaGQxxAPI4dXc_shUrjROj7OrKt?usp=share_link

#### XLM-RoBERTa-base
https://drive.google.com/drive/folders/1CTgIEIDNHhV75qQ7-uovt6oXkiUIAVH8?usp=share_link

#### XLM-RoBERTa-large
https://drive.google.com/drive/folders/1BzdVmN51f33NHrdemJajz67MmlZljB2J?usp=share_link



## Code

To fine-tune the model please run the file `train_model.py`.
It expects two arguments:
- Model: **distilbert** or **xlmroberta-base** or **xlmroberta-large**
- Loss function: **mse** or **ccc** or **robust** or **mse+ccc** or **robust+ccc**

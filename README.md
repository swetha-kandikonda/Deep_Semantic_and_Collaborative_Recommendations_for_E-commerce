# Deep Semantic & Collaborative Recommendations for E-commerce

## Abstract
This study aims to address the persistent limitations of traditional recommendation systems in e-commerce, namely data sparsity and the cold-start problem. The primary research question is how to design an effective hybrid model that synergistically combines collaborative filtering with advanced natural language processing to deliver relevant, personalized product recommendations for all users, including those with limited interaction history. We developed a hybrid recommendation system integrating a GridSearchCV-optimized Singular Value Decomposition (SVD) model to capture latent patterns in user-item interaction data and a content-based model using pre-trained Sentence-BERT to generate deep semantic embeddings from product text. Using the Amazon Video Games dataset, which was filtered, preprocessed, and temporally split, the optimized SVD model achieved a final cross-validation RMSE of 0.991. The complete hybrid model demonstrated strong performance on the test set, achieving a $Precision@10$ of 13.33%, a $Recall@10$ of 32.33%, $F1@10$ of 16.06 and an $NDCG@10$ of 21.36%. Qualitative analysis further confirmed the model's ability to generate logical, genre-specific recommendations for cold-start users. The proposed hybrid approach successfully mitigates the cold-start problem and enhances overall recommendation relevance by fusing user behavior patterns with deep semantic understanding, providing a robust framework for creating more personalized e-commerce experiences.

**Keywords:** Hybrid Recommendation System, Large Language Models (LLM), Sentence-BERT, Collaborative Filtering, SVD, Personalization, E-Commerce, Cold-Start Problem, Amazon Video Games.


## Introduction

In the modern digital marketplace, e-commerce platforms face the dual challenge of vast inventories and short user attention spans. As product catalogs grow, consumers can experience the "paradox of choice," leading to indecision. Effective product recommendation systems have become indispensable for enhancing the user experience, creating personalized discovery pathways, and driving key business metrics like engagement, conversion rates, and customer loyalty.

The foundational approaches are Collaborative Filtering (CF), which leverages similar users' behavior, and Content-Based Filtering (CBF), which uses item attributes. However, these methods suffer from data sparsity, scalability, and the cold-start problem for new users or items. Early NLP techniques like TF-IDF used in these systems often failed to capture deep semantic nuances in product descriptions.

To overcome these challenges, Matrix Factorization (MF) techniques like Singular Value Decomposition (SVD) were introduced to learn latent factors from sparse user-item matrices. More recently, deep learning, including Autoencoders and Graph Neural Networks (GNNs), has been used to capture complex, non-linear relationships.

The evaluation of recommendation systems has also evolved. While offline metrics like Root Mean Squared Error (RMSE) measure rating prediction accuracy, they don't always correlate with user satisfaction. This has led to a shift towards ranking-based metrics like $Precision@k$, $Recall@k$, and Normalized Discounted Cumulative Gain ($NDCG@k$), which measure the quality of top-ranked items. Researchers also now emphasize "beyond-accuracy" metrics such as novelty, diversity, and serendipity.

Our project addresses the existing research gap by proposing a hybrid filtering system that combines the strengths of SVD for modeling user-item interactions with the advanced NLP of Sentence-BERT for a deep semantic understanding of product text. This synergistic design allows our model to overcome the cold-start problem and provide robust, context-aware recommendations.

## Methodology

Our methodology involved designing an end-to-end pipeline to transform raw data into personalized recommendations. The process began with exploring collaborative filtering (CF) methods like SVD, which are effective at uncovering latent factors but suffer from the cold-start problem, data sparsity, and popularity bias.

To address these limitations, we adopted a hybrid framework incorporating content-based filtering (CBF) to integrate item attributes with interaction patterns. This approach helps mitigate cold-start scenarios and improve personalization.


![Figure 1: Architecture of a Hybrid Recommendation System](./figs/Architecture%20of%20a%20Hybrid%20Recommendation%20System.png)
*Figure 1: Architecture of a Hybrid Recommendation System*



![Figure 2: Hybrid Recommendation System Employing a Hybridization Module](./figs/Hybrid%20Recommendation%20System%20Employing%20a%20Hybridization%20Module.png)
*Figure 2: Hybrid Recommendation System Employing a Hybridization Module*

The Amazon Video Games dataset used lacked detailed structured features (e.g., genre, platform), limiting traditional CBF methods. To overcome this, we used Large Language Models (LLMs), specifically Sentence-BERT, to transform review text and metadata into dense semantic embeddings. These embeddings serve as rich proxy product descriptions, enabling the system to model relationships between items even without structured features.

![Figure 3: Enhancing Item Representation using LLMs](./figs/Enhancing%20Item%20Representation%20using%20Large%20Language%20Models%20(LLMs).png)
*Figure 3: Enhancing Item Representation using Large Language Models (LLMs)*

### End-to-End Workflow

The system architecture follows a six-step process from data collection to analysis.

![Figure 4: End-to-End Recommendation System Workflow](/figs/End-to-End%20Recommendation%20System%20Workflow.png)
*Figure 4: End-to-End Recommendation System Workflow*

#### 1. Data Collection & Preprocessing
* **Dataset:** Amazon Video Games Dataset.
* **Filtering:** The initial dataset of 73,042 reviews was filtered to a core sample of 14,068 interactions by including only users with at least five reviews and products with at least ten reviews.
* **Temporal Split:** The data was split chronologically to simulate a real-world scenario, using older data for training and recent data for testing.

#### 2. Exploratory Data Analysis (EDA)
An EDA was performed to understand the dataset's characteristics.
* **Ratings Distribution:** The ratings showed a strong positive skew towards 4 and 5 stars, indicating a generally satisfied user base. This motivated the use of ranking-based metrics over simple accuracy.

    ![Figure 5: Distribution of Ratings](./figs/Analysis%20of%20User%20Rating%20Frequency.png)
    *Figure 5: Analysis of User Rating Frequency*

* **User Activity:** The analysis revealed a "long-tail" distribution, where a few "power users" contribute a large number of reviews. This confirmed the presence of data sparsity and justified our data filtering strategy.

    ![Figure 6: Long-Tail Distribution of User Activity](./figs/Long-Tail%20Distribution%20of%20User%20Review%20Activity.png)
    *Figure 6: Long-Tail Distribution of User Review Activity*

#### 3. Content-Based Filtering (Sentence-BERT)
* The content-based component used the pre-trained `all-MiniLM-L6-v2` Sentence-BERT model.
* A new text feature was created by concatenating each product's title, summary, and description, which was then transformed into a 384-dimensional vector.
* t-SNE visualization confirmed that these embeddings effectively clustered games of similar genres.

    ![Figure 7: t-SNE Visualization of Product Embeddings](./figs/2D%20Projection%20of%20Product%20Embeddings%20Using%20t-SNE.png)
    *Figure 7: 2D Projection of Product Embeddings Using t-SNE*

#### 4. Collaborative Filtering (SVD)
* The collaborative filtering component was built using a Singular Value Decomposition (SVD) model.
* A `GridSearchCV` was performed to find the optimal hyperparameters: `n_factors: 10`, `n_epochs: 30`, `lr_all: 0.005`, and `reg_all: 0.05`.
* The tuned SVD model achieved a final cross-validation RMSE of 0.991.

#### 5. Hybrid Model
* The final model integrates the SVD and Sentence-BERT outputs using a weighted average, with a weight of $\alpha=0.6$ for the collaborative score.
* To address the cold-start problem, the model defaults to a purely content-based approach for users with fewer than three ratings.

## Evaluation & Results

The hybrid model was evaluated on a test set of 27 users for its ability to rank relevant items in the top 10 recommendations.

### Quantitative Metrics

The model demonstrated strong performance across several ranking-based metrics.

| Metric | Score |
| :--- | :--- |
| **Precision@10** | 13.33% |
| **Recall@10** | 32.33% |
| **F1@10** | 16.06% |
| **NDCG@10** | 21.36% |

![Figure 8: Hybrid Model Evaluation Metrics](/figs/Hybrid%20Recommendation%20System%20Employing%20a%20Hybridization%20Module.png)
*Figure 8: Evaluation Metrics for the Hybrid Recommendation Model*

### Qualitative Analysis

A qualitative analysis was conducted to assess the model's practical behavior. For a user who enjoyed fighting games, the system logically recommended other highly-rated games in the same genre, such as *Marvel Vs. Capcom 2* and *Super Smash Bros Melee*, demonstrating its ability to capture nuanced genre preferences.

![Figure 9: User Recommendation Profile](./figs/User%20Recommendation%20Profile%20Analysis.png)
*Figure 9: User Recommendation Profile Analysis*

![Figure 10: Example Recommendations](https://i.imgur.com/your-image-link-for-figure10.png)
*Figure 10: Example of Recommended Games for a Specific User*

## Conclusion

We successfully designed and evaluated a sophisticated hybrid recommendation system by integrating an optimized SVD-based collaborative filter with a state-of-the-art Sentence-BERT language model. The model effectively mitigates the cold-start problem and generates highly relevant recommendations, validated by strong quantitative metrics ($Precision@10$ of 13.33%, $Recall@10$ of 32.33%, and $NDCG@10$ of 21.36%). This confirms the power of combining traditional and deep learning techniques to create a more engaging user experience.

## Future Work
Several avenues exist for future work:
* **Integrate Advanced LLMs:** Use more advanced Large Language Models to enhance the semantic understanding of products and user satisfaction.
* **Beyond-Accuracy Metrics:** Broaden evaluation to include metrics like novelty, diversity, and serendipity.
* **Real-Time Processing:** Develop a session-based recommendation model using Recurrent Neural Networks (RNNs) to adapt to a user's immediate actions.

## Conflict of Interest
On behalf of all authors, the corresponding author states that there is no conflict of interest.

## References

> [1] Shukla, A., Khare, R., & Shukla, A. (2020). *Online Ecommerce Hypermarket Shopping Site with Product Recommendation System, Prediction based System by Machine Learning, Internet Security and Artificial Intelligence*. International Journal of Advance Study and Research Work (IJASRW), 3(10), 14-20.
>
> [2] Addagarla, S. K., & Amalanathan, A. (2020). *Probabilistic Unsupervised Machine Learning for a Similar Image Recommender System for E-Commerce*. Symmetry, 12(11), 1783.
>
> [3] Hussien, F. T. A., Rahma, A. M. S., & Abdul Wahab, H. B. (2021). *Recommendation Systems for E-commerce Systems: An Overview*. Journal of Physics: Conference Series, 1897(1), 012024.
>
> [4] Tahir, M., Enam, R. N., & Mustafa, S. M. N. (2021). *E-commerce platform based on Machine Learning Recommendation System*. 2021 6th International Multi-Topic ICT Conference (IMTIC), 1-6.
>
> [5] Addagarla, S. K., & Amalanathan, A. (2021). *e-SimNet: A visual similar product recommender system for E-commerce*. Indonesian Journal of Electrical Engineering and Computer Science, 22(1), 563-570.
>
> [6] Naz, A., Khan, U. A., Sodhar, I. H., & Buller, A. H. (2022). *Product Recommendation Using Machine Learning: A Review of Existing Techniques*. International Journal of Computer Science and Network Security (IJCSNS), 22(5), 72-80.
>
> [7] Rahman, A., Haque, Z., & Ahammad, M. S. (2024). *E-Commerce Product Recommendation System Using Machine Learning Algorithms*. International Journal of Computer Science and Information Security (IJCSIS), 22(3), 1-8.
>
> [8] Shankar, A., Perumal, P., Subramanian, M., Ramu, N., Natesan, D., Kulkarni, V. R., & Stephan, T. (2024). *An intelligent recommendation system in e-commerce using ensemble learning*. Multimedia Tools and Applications, 83(11), 48521-48537.

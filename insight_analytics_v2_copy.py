"""
Chatbot Query Analytics Dashboard - v6
LLM-powered Investment Theme Analysis using LangChain.

Changes from v5:
- Removed answer_found filter widget and all related filtering logic.
- Removed Query Status Breakdown chart (answered vs unanswered stacked bar)
  from Tab 1, as it relied on answer_found for colouring.
- KPI row simplified to Total Queries and Answer Rate only; answered /
  unanswered counts removed since the filter is gone.
- answer_found column is still fetched from the DB (used by Tab 4 Content
  Gap Analysis to isolate unanswered queries) but is no longer exposed as
  a user-facing filter.
- Inline filter row reduced from three columns to two (From / To dates).
"""

import streamlit as st
import pandas as pd
import sqlite3
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from datetime import datetime
import re
from collections import Counter
import numpy as np
import json
import os
from typing import List, Dict

# LangChain / OpenAI
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# Traditional NLP (kept for word cloud + LDA fallback)
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.tokenize import word_tokenize
from dotenv import load_dotenv
load_dotenv()

# ── NLTK bootstrapping ────────────────────────────────────────────────────────
for resource, path in [
    ('punkt',     'tokenizers/punkt'),
    ('punkt_tab', 'tokenizers/punkt_tab'),
    ('stopwords', 'corpora/stopwords'),
]:
    try:
        nltk.data.find(path)
    except LookupError:
        nltk.download(resource, quiet=True)


# ═════════════════════════════════════════════════════════════════════════════
#  INVESTMENT THEME ANALYSER  (LangChain-powered)
# ═════════════════════════════════════════════════════════════════════════════

INVESTMENT_ANALYST_SYSTEM = """You are a first class Senior Investment Analyst and Portfolio Strategist with deep expertise in equity markets, fixed income, macro-economics, and alternative assets.

Your task is to analyse a batch of investor/analyst queries and synthesize them through the following lens:

1. Group into **Macro Themes**: Use broad buckets (e.g., Monetary Policy, AI & Technology, Geopolitical Risk, Credit Markets). 

2. Identify **Specific Topics (Consolidation Rule)**: Within each Macro Theme, you must merge queries that share the same underlying driver. 
   * *Critical:* If multiple queries refer to the same catalyst (e.g., "Fed hikes," "Dot plot," and "Higher for longer"), group them into ONE Specific Topic rather than listing them separately. 
   * Aim for "Topic Density"—fewer, more robust groups are better than many thin ones.

3. Provide a concise **Analyst Note**: A 1-2 sentence briefing on the investment implication or "bottom line" for a portfolio manager.

Return ONLY a JSON array. Each element must have exactly these keys:
- "query":          the original query text (string)
- "macro_theme":    high-level investment theme (string)
- "specific_topic": precise sub-topic (string)
- "analyst_note":   brief investment implication (string)

No markdown fences, no extra keys, no preamble."""

INVESTMENT_ANALYST_USER = """Analyse the following {n} investor queries and return the JSON array as instructed.

Queries:
{queries}"""


class InvestmentThemeAnalyser:
    """
    LangChain-powered analyst that classifies investor queries into
    structured investment themes using an LLM.
    """

    def __init__(self, openai_api_key: str, model: str = "gpt-4o-mini"):
        self.llm = ChatOpenAI(
            model=model,
            temperature=0,
            openai_api_key=openai_api_key,
        )
        self.embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

        self.prompt = ChatPromptTemplate.from_messages([
            ("system", INVESTMENT_ANALYST_SYSTEM),
            ("human",  INVESTMENT_ANALYST_USER),
        ])
        self.chain = self.prompt | self.llm


    # ── semantic deduplication ────────────────────────────────────────────────
    def _semantic_deduplicate(
        self,
        queries: list[str],
        max_queries: int = 60,
        similarity_threshold: float = 0.92,
    ) -> list[str]:
        """
        Remove near-duplicate queries using cosine similarity on embeddings.
        Returns at most `max_queries` representative queries.
        Designed to cut LLM token costs while preserving semantic coverage.
        """
        unique = list(dict.fromkeys(q.strip() for q in queries if q.strip()))
        if len(unique) <= max_queries:
            return unique[:max_queries]

        try:
            vecs = np.array(self.embeddings.embed_documents(unique))
            kept_indices = []
            for i, vec in enumerate(vecs):
                if not kept_indices:
                    kept_indices.append(i)
                    continue
                sims = cosine_similarity([vec], vecs[kept_indices])[0]
                if sims.max() < similarity_threshold:
                    kept_indices.append(i)
                if len(kept_indices) >= max_queries:
                    break
            return [unique[i] for i in kept_indices]
        except Exception:
            return unique[:max_queries]

    # ── core LLM call ─────────────────────────────────────────────────────────
    def classify(self, queries: list[str]) -> pd.DataFrame:
        """
        Classify a list of query strings into investment themes.
        Returns a DataFrame with columns:
          query | macro_theme | specific_topic | analyst_note
        """
        deduped = self._semantic_deduplicate(queries)
        if not deduped:
            return pd.DataFrame()

        batch_size = 30
        all_results: list[dict] = []

        for start in range(0, len(deduped), batch_size):
            batch = deduped[start : start + batch_size]
            numbered = "\n".join(f"{i+1}. {q}" for i, q in enumerate(batch))

            try:
                response = self.chain.invoke({
                    "n":       len(batch),
                    "queries": numbered,
                })
                raw = response.content.strip()

                # Strip accidental markdown fences
                if raw.startswith("```"):
                    raw = re.sub(r"^```[a-zA-Z]*\n?", "", raw)
                    raw = re.sub(r"\n?```$", "", raw)

                parsed = json.loads(raw)
                if isinstance(parsed, list):
                    all_results.extend(parsed)
                elif isinstance(parsed, dict):
                    for v in parsed.values():
                        if isinstance(v, list):
                            all_results.extend(v)
                            break
            except json.JSONDecodeError as e:
                st.warning(f"⚠️ LLM returned non-JSON for batch {start//batch_size + 1}: {e}")
            except Exception as e:
                st.error(f"❌ LLM call failed for batch {start//batch_size + 1}: {e}")

        if not all_results:
            return pd.DataFrame()

        df = pd.DataFrame(all_results)
        df.columns = [c.lower().replace(" ", "_") for c in df.columns]
        expected = ["query", "macro_theme", "specific_topic", "analyst_note"]
        for col in expected:
            if col not in df.columns:
                df[col] = "Unknown"
        return df[expected]

    # ── aggregate insights ────────────────────────────────────────────────────
    @staticmethod
    def aggregate(classified_df: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate classified queries into a ranked theme summary.
        Returns: macro_theme | specific_topic | mentions | sample_note
        """
        if classified_df.empty:
            return pd.DataFrame()

        agg = (
            classified_df
            .groupby(["macro_theme", "specific_topic"])
            .agg(
                mentions    = ("query",        "count"),
                sample_note = ("analyst_note", "first"),
            )
            .reset_index()
            .sort_values("mentions", ascending=False)
        )
        return agg


# ═════════════════════════════════════════════════════════════════════════════
#  QUERY ANALYTICS  (data + traditional NLP)
# ═════════════════════════════════════════════════════════════════════════════

class QueryAnalytics:
    """Handles data loading, preprocessing, word cloud, and LDA fallback."""

    def __init__(self, db_path: str = "query_analytics.db"):
        self.db_path = db_path
        self.df = None
        self._init_database()
        self.load_data()

    
    def _init_database(self):
        """Initialize SQLite database with query tracking tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Main queries table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS queries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                query_text TEXT NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                answer_found BOOLEAN,
                relevance_score REAL,
                sources TEXT,
                response_length INTEGER,
                session_id TEXT,
                user_id TEXT
            )
        ''')
        
        # Topics extracted from queries
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS query_topics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                query_id INTEGER,
                topic TEXT,
                FOREIGN KEY (query_id) REFERENCES queries(id)
            )
        ''')
        
        # Create indexes for faster queries
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_timestamp 
            ON queries(timestamp)
        ''')
        
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_topic 
            ON query_topics(topic)
        ''')
        
        conn.commit()
        conn.close()
    
    def log_query(self, query_text: str, answer_found: bool, 
                  relevance_score: float, sources: List[str],
                  response_length: int, session_id: str = None,
                  user_id: str = None):
        """
        Log a user query with metadata.
        
        Args:
            query_text: The user's question
            answer_found: Whether an answer was found
            relevance_score: Highest relevance score from vector search
            sources: List of source documents used
            response_length: Length of the response in characters
            session_id: Session identifier
            user_id: User identifier (optional)
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Insert query
        cursor.execute('''
            INSERT INTO queries 
            (query_text, answer_found, relevance_score, sources, 
             response_length, session_id, user_id)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            query_text,
            answer_found,
            relevance_score,
            json.dumps(sources),
            response_length,
            session_id,
            user_id
        ))
        
        query_id = cursor.lastrowid
        
        # Extract and store topics
        topics = self._extract_topics(query_text)
        for topic in topics:
            cursor.execute('''
                INSERT INTO query_topics (query_id, topic)
                VALUES (?, ?)
            ''', (query_id, topic))
        
        conn.commit()
        conn.close()
    

    def _extract_topics(self, query_text: str) -> List[str]:
        """
        Extract key topics/keywords from query text.
        Simple implementation - can be enhanced with NLP.
        """
        # Remove common words
        stop_words = {
            'what', 'is', 'are', 'the', 'a', 'an', 'how', 'why', 'when',
            'where', 'who', 'can', 'you', 'tell', 'me', 'about', 'for',
            'in', 'on', 'at', 'to', 'from', 'with', 'of', 'and', 'or', 'provide', 'detail', 'that',
            'provide', 'the', 'latest', 'outlook', 'please'
        }
        
        # Tokenize and clean
        words = re.findall(r'\b[a-z]{3,}\b', query_text.lower())
        topics = [w for w in words if w not in stop_words]
        
        # Also extract multi-word phrases (bigrams)
        bigrams = []
        for i in range(len(words) - 1):
            if words[i] not in stop_words or words[i+1] not in stop_words:
                bigram = f"{words[i]} {words[i+1]}"
                bigrams.append(bigram)
        
        return list(set(topics + bigrams[:5]))  # Limit bigrams
    
    # ── data loading ──────────────────────────────────────────────────────────
    def load_data(self):
        conn = sqlite3.connect(self.db_path)
        query = """
        SELECT id, query_text, timestamp, answer_found,
               relevance_score, sources, response_length, session_id, user_id
        FROM queries
        ORDER BY timestamp DESC
        """
        self.df = pd.read_sql_query(query, conn)
        self.df['timestamp'] = pd.to_datetime(self.df['timestamp'])
        conn.close()

    def load_json_data(self, json_file: str):
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            return None

    # ── text preprocessing ────────────────────────────────────────────────────
    def preprocess_text(self, text: str) -> str:
        if pd.isna(text):
            return ""
        text = text.lower()
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        tokens = word_tokenize(text)

        stopwords_json = self.load_json_data("stopwords.json")
        if stopwords_json and "stopwords" in stopwords_json:
            stop_words = set(stopwords_json["stopwords"])
        else:
            from nltk.corpus import stopwords as nltk_sw
            stop_words = set(nltk_sw.words('english'))
            stop_words.update({'what', 'why', 'how', 'when', 'where',
                                'provide', 'tell', 'show', 'explain', 'give'})

        return ' '.join(w for w in tokens if w not in stop_words and len(w) > 2)

    # ── word frequencies (word cloud) ─────────────────────────────────────────
    def get_word_frequencies(self, texts) -> Counter:
        all_text = ' '.join(self.preprocess_text(t) for t in texts)
        return Counter(all_text.split())

    # ── LDA fallback ──────────────────────────────────────────────────────────
    def extract_topics_lda(self, texts: list, n_topics: int = 5, n_words: int = 5) -> dict:
        """LDA-based topic extraction — used as fallback when LLM is unavailable."""
        processed = [self.preprocess_text(t) for t in texts if t]
        processed = [t for t in processed if t]
        if not processed:
            return {}

        vectorizer = CountVectorizer(
            ngram_range=(1, 2), max_features=1000,
            stop_words='english', min_df=2
        )
        try:
            dtm = vectorizer.fit_transform(processed)
        except ValueError:
            return {}

        lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
        lda.fit(dtm)
        feature_names = vectorizer.get_feature_names_out()
        topics = {}
        for idx, topic in enumerate(lda.components_):
            top_words = [feature_names[i] for i in topic.argsort()[-n_words:][::-1]]
            topics[f"Theme Group {idx + 1}"] = top_words
        return topics


# ═════════════════════════════════════════════════════════════════════════════
#  DASHBOARD
# ═════════════════════════════════════════════════════════════════════════════

def create_dashboard(db_path: str):
    st.set_page_config(
        page_title="Chatbot Analytics Dashboard",
        page_icon="📊",
        layout="wide",
    )

    # ── API key and model read from environment only ───────────────────────────
    # This is for local
    for key, value in st.secrets.items():
        os.environ[key] = value
    api_key   = os.environ.get("OPENAI_API_KEY", "")
    llm_model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")


    st.title("📊 Chatbot Query Analytics Dashboard")
    st.markdown("---")

    # ── load full dataset ──────────────────────────────────────────────────────
    analytics = QueryAnalytics(db_path)
    df = analytics.df

    # ── inline date range filter ──────────────────────────────────────────────
    min_date = df['timestamp'].min().date()
    max_date = df['timestamp'].max().date()

    filter_col1, filter_col2 = st.columns([2, 2])

    with filter_col1:
        start_date = st.date_input(
            "From",
            value=min_date,
            min_value=min_date,
            max_value=max_date,
            key="filter_start",
        )

    with filter_col2:
        end_date = st.date_input(
            "To",
            value=max_date,
            min_value=min_date,
            max_value=max_date,
            key="filter_end",
        )

    # Guard: ensure start is not after end
    if start_date > end_date:
        st.error("⚠️ 'From' date must be on or before 'To' date.")
        st.stop()

    # Apply date range filter
    mask = (
        (df['timestamp'].dt.date >= start_date) &
        (df['timestamp'].dt.date <= end_date)
    )
    filtered_df = df[mask].copy()

    st.markdown("---")

    # ── top-level KPIs ────────────────────────────────────────────────────────
    answered    = int(filtered_df['answer_found'].sum())
    answer_rate = answered / len(filtered_df) * 100 if len(filtered_df) else 0

    kpi1, kpi2 = st.columns(2)
    kpi1.metric("Total Queries", len(filtered_df))
    kpi2.metric("Answer Rate",   f"{answer_rate:.1f}%")
    st.markdown("---")

    # ── tabs ──────────────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4 = st.tabs([
        "📈 Prompt Frequencies",
        "☁️ Word Cloud & LDA Topics",
        "🏦 Investment Theme Intelligence",
        "🔍 Content Gap Analysis",
    ])

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 1 — PROMPT FREQUENCIES
    # ══════════════════════════════════════════════════════════════════════════
    with tab1:
        st.header("Prompt Frequencies Over Time")

        granularity = st.selectbox(
            "Time Granularity",
            ["Hour", "Day", "Week", "Month"],
            index=1,
        )

        df_freq = filtered_df.copy()
        if granularity == "Hour":
            df_freq['period'] = df_freq['timestamp'].dt.floor('h')
        elif granularity == "Day":
            df_freq['period'] = df_freq['timestamp'].dt.date
        elif granularity == "Week":
            df_freq['period'] = df_freq['timestamp'].dt.to_period('W').apply(lambda r: r.start_time)
        else:
            df_freq['period'] = df_freq['timestamp'].dt.to_period('M').apply(lambda r: r.start_time)

        freq_data = df_freq.groupby('period').size().reset_index(name='count')

        fig_freq = px.line(
            freq_data, x='period', y='count',
            title=f'Queries per {granularity}',
            labels={'period': 'Period', 'count': 'Queries'},
            markers=True,
        )
        fig_freq.update_layout(hovermode='x unified')
        st.plotly_chart(fig_freq, use_container_width=True)

        if not freq_data.empty:
            peak_row = freq_data.loc[freq_data['count'].idxmax()]
            col1, col2 = st.columns(2)
            col1.metric("Avg Queries per Period", f"{freq_data['count'].mean():.1f}")
            col2.metric("Peak Period", str(peak_row['period']), f"{peak_row['count']} queries")

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 2 — WORD CLOUD & LDA
    # ══════════════════════════════════════════════════════════════════════════
    with tab2:
        st.header("Word Cloud & LDA Topic Groups")
        st.info("This tab shows keyword-level signals. For investment-grade theme analysis, see the **🏦 Investment Theme Intelligence** tab.")

        col_wc, col_lda = st.columns([1, 1])

        with col_wc:
            st.subheader("Keyword Cloud")
            word_freq = analytics.get_word_frequencies(filtered_df['query_text'])
            if word_freq:
                wc = WordCloud(
                    width=700, height=400,
                    background_color='white',
                    colormap='viridis',
                    max_words=100,
                ).generate_from_frequencies(word_freq)
                fig_wc, ax = plt.subplots(figsize=(7, 4))
                ax.imshow(wc, interpolation='bilinear')
                ax.axis('off')
                st.pyplot(fig_wc)
                plt.close(fig_wc)
            else:
                st.info("No data available for word cloud.")

        with col_lda:
            st.subheader("LDA Thematic Groups (Fallback NLP)")
            n_topics = st.slider("Number of LDA Groups", 3, 10, 5, key="lda_slider")
            topics = analytics.extract_topics_lda(
                filtered_df['query_text'].tolist(), n_topics=n_topics
            )
            if topics:
                for t_name, words in topics.items():
                    with st.expander(f"📁 {t_name}"):
                        st.write(f"**Terms:** {', '.join(words)}")
            else:
                st.info("Need more data for LDA grouping.")

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 3 — LLM INVESTMENT THEME INTELLIGENCE
    # ══════════════════════════════════════════════════════════════════════════
    with tab3:
        st.header("🏦 LLM-Powered Investment Theme Intelligence")
        st.markdown(
            "An LLM acting as a **Senior Investment Analyst** reads every query and assigns "
            "it to a *Macro Theme* and *Specific Topic*."
        )

        if not api_key:
            st.warning(
                "⚠️ No OpenAI API key detected. "
                "Set the **OPENAI_API_KEY** environment variable to enable LLM-powered analysis."
            )
            st.stop()

        # ── run / cache ────────────────────────────────────────────────────────
        cache_key = f"llm_themes_{hash(tuple(filtered_df['query_text'].tolist()))}"

        if st.button("🔄 Run / Refresh Investment Theme Analysis", type="primary"):
            st.session_state.pop(cache_key, None)

        if cache_key not in st.session_state:
            with st.spinner("🤖 Analysing queries with LLM investment analyst…"):
                try:
                    analyser = InvestmentThemeAnalyser(
                        openai_api_key=api_key,
                        model=llm_model,
                    )
                    classified_df = analyser.classify(
                        filtered_df['query_text'].tolist()
                    )
                    st.session_state[cache_key] = classified_df
                except Exception as e:
                    st.error(f"LLM analysis failed: {e}")
                    st.session_state[cache_key] = pd.DataFrame()

        classified_df: pd.DataFrame = st.session_state.get(cache_key, pd.DataFrame())

        if classified_df.empty:
            st.info("No classification results yet. Click **Run Analysis** above.")
            st.stop()

        aggregated_df = InvestmentThemeAnalyser.aggregate(classified_df)

        # ── KPIs ───────────────────────────────────────────────────────────────
        n_macro  = classified_df['macro_theme'].nunique()
        n_topics = classified_df['specific_topic'].nunique()
        k1, k2   = st.columns(2)
        k1.metric("Macro Themes Identified", n_macro)
        k2.metric("Specific Topics",         n_topics)
        st.markdown("---")

        # ── macro theme treemap ────────────────────────────────────────────────
        st.subheader("📊 Macro Theme Distribution")
        macro_counts = (
            classified_df
            .groupby(['macro_theme', 'specific_topic'])
            .size().reset_index(name='count')
        )

        # Compute a normalized intensity per macro_theme for richer coloring
        total = macro_counts['count'].sum()

        fig_tree = px.treemap(
            macro_counts,
            path=['macro_theme', 'specific_topic'],
            values='count',
            title="Investment Theme Treemap — area proportional to query volume",
        )

        fig_tree.update_traces(
            texttemplate=(
                "<b>%{label}</b><br>"
                "%{value} queries<br>"
                "<span style='font-size:11px;opacity:0.8'>%{percentRoot:.1%} of total</span>"
            ),
            textposition="middle center",
            textfont=dict(
                family="'IBM Plex Sans', 'Helvetica Neue', sans-serif",
                size=13,
                color="white",
            ),
            marker=dict(
                pad=dict(t=28, l=6, r=6, b=6),         # breathing room inside each cell
            ),
            hovertemplate=(
                "<b>%{label}</b><br>"
                "Queries: <b>%{value}</b><br>"
                "Share of total: <b>%{percentRoot:.1%}</b><br>"
                "<extra></extra>"
            ),
            root_color="#0a0f1a",
        )

        fig_tree.update_layout(
            title=dict(
                text="Investment Theme Treemap — area proportional to query volume",
                font=dict(
                    family="'IBM Plex Sans', sans-serif",
                    size=17,
                    color="#1a2940",
                ),
                x=0.01,
                xanchor="left",
            ),
            margin=dict(t=55, l=4, r=4, b=4),
            paper_bgcolor="#f7f9fc",
            coloraxis_showscale=True,
            coloraxis_colorbar=dict(
                title=dict(text="Query<br>Volume", font=dict(size=11)),
                thickness=14,
                len=0.6,
                tickfont=dict(size=10),
                outlinewidth=0,
                bgcolor="rgba(247,249,252,0.8)",
            ),
        )

        st.plotly_chart(fig_tree, use_container_width=True)

        st.markdown("---")

        # ── ranked theme table with analyst notes ──────────────────────────────
        st.subheader("📋 Ranked Investment Themes — Analyst Summary")

        macro_groups = (
            aggregated_df
            .groupby("macro_theme", sort=False)
            .apply(lambda g: g.sort_values("mentions", ascending=False))
            .reset_index(drop=True)
        )

        for macro_theme, topic_rows in macro_groups.groupby("macro_theme", sort=False):
            total_mentions = topic_rows["mentions"].sum()
            n_topics_count = len(topic_rows)

            with st.expander(
                f"🏦 **{macro_theme}** — {total_mentions} "
                f"{'query' if total_mentions == 1 else 'queries'} · {n_topics_count} "
                f"{'topic' if n_topics_count == 1 else 'topics'}"
            ):

                for _, row in topic_rows.iterrows():
                    with st.expander(
                        f"🏷️ {row['specific_topic']} "
                        f"— {row['mentions']} {'query' if row['mentions'] == 1 else 'queries'}"
                    ):
                        st.markdown(f"**Mentions:** {row['mentions']}")
                        st.markdown(f"**Analyst Note:** {row['sample_note']}")

                        mask = (
                            (classified_df["macro_theme"]   == row["macro_theme"]) &
                            (classified_df["specific_topic"] == row["specific_topic"])
                        )
                        sample_queries = classified_df.loc[mask, "query"].head(5).tolist()
                        if sample_queries:
                            st.markdown("**Sample Queries:**")
                            for q in sample_queries:
                                st.markdown(f"- _{q}_")

        st.markdown("---")

        # ── raw classified data download ───────────────────────────────────────
        st.subheader("📥 Download Classified Query Data")
        csv_classified = classified_df.to_csv(index=False)
        st.download_button(
            label="Download Full Classification CSV",
            data=csv_classified,
            file_name=f"llm_investment_themes_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv",
        )

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 4 — CONTENT GAP ANALYSIS
    # ══════════════════════════════════════════════════════════════════════════
    with tab4:
        st.header("Content Gap Analysis")
        st.markdown("*Unanswered queries reveal content gaps in the knowledge base.*")

        unanswered_df = filtered_df[filtered_df['answer_found'] == False].copy()

        if len(unanswered_df) == 0:
            st.success("🎉 No unanswered queries in the selected date range.")
            st.balloons()
        else:
            c1, c2, c3 = st.columns(3)
            c1.metric("Total Unanswered",    len(unanswered_df))
            c2.metric("Affected Sessions",   unanswered_df['session_id'].nunique())
            c3.metric("Avg Relevance Score", f"{unanswered_df['relevance_score'].mean():.2f}")
            st.markdown("---")

            # LLM gap analysis (if API key available)
            if api_key and len(unanswered_df) >= 3:
                st.subheader("🤖 LLM-Identified Gaps in Investment Coverage")
                gap_cache_key = f"gap_themes_{hash(tuple(unanswered_df['query_text'].tolist()))}"

                if st.button("🔄 Run Gap Theme Analysis", key="gap_btn"):
                    st.session_state.pop(gap_cache_key, None)

                if gap_cache_key not in st.session_state:
                    with st.spinner("Analysing unanswered queries with LLM…"):
                        try:
                            analyser = InvestmentThemeAnalyser(
                                openai_api_key=api_key,
                                model=llm_model,
                            )
                            gap_df = analyser.classify(unanswered_df['query_text'].tolist())
                            st.session_state[gap_cache_key] = gap_df
                        except Exception as e:
                            st.error(f"Gap analysis failed: {e}")
                            st.session_state[gap_cache_key] = pd.DataFrame()

                gap_df: pd.DataFrame = st.session_state.get(gap_cache_key, pd.DataFrame())

                if not gap_df.empty:
                    gap_agg = InvestmentThemeAnalyser.aggregate(gap_df)
                    st.markdown("**Top unanswered investment themes (knowledge base gaps):**")
                    for _, row in gap_agg.head(10).iterrows():
                        with st.expander(
                            f"⚠️ **{row['macro_theme']}** › {row['specific_topic']} "
                            f"— {row['mentions']} unanswered {'query' if row['mentions'] == 1 else 'queries'}"
                        ):
                            st.write(f"**Analyst Note:** {row['sample_note']}")
                else:
                    st.info("Click **Run Gap Theme Analysis** to identify content gaps.")

            elif len(unanswered_df) >= 3:
                st.subheader("🏷️ LDA Topics in Unanswered Queries (Fallback)")
                n_gap_topics = min(5, len(unanswered_df) // 2)
                gap_topics = analytics.extract_topics_lda(
                    unanswered_df['query_text'].tolist(),
                    n_topics=n_gap_topics, n_words=8,
                )
                for t_name, words in gap_topics.items():
                    with st.expander(f"**{t_name}**: {', '.join(words[:5])}…"):
                        st.write("**Key Terms:**", ', '.join(words))
                        matches = [
                            r['query_text'] for _, r in unanswered_df.iterrows()
                            if any(w in r['query_text'].lower() for w in words[:5])
                        ]
                        if matches:
                            st.write("**Sample Queries:**")
                            for q in matches[:3]:
                                st.write(f"- {q.strip()}")
            else:
                st.info("Need at least 3 unanswered queries for gap analysis.")

            st.markdown("---")
            st.subheader("📋 Individual Unanswered Queries")

            sort_by = st.selectbox(
                "Sort by",
                ["Most Recent", "Relevance Score", "Query Length"],
            )
            if sort_by == "Most Recent":
                disp = unanswered_df.sort_values('timestamp', ascending=False)
            elif sort_by == "Relevance Score":
                disp = unanswered_df.sort_values('relevance_score', ascending=False)
            else:
                unanswered_df['query_length'] = unanswered_df['query_text'].str.len()
                disp = unanswered_df.sort_values('query_length', ascending=False)

            disp_table = disp[['timestamp', 'query_text', 'relevance_score']].copy()
            disp_table['timestamp'] = disp_table['timestamp'].dt.strftime('%Y-%m-%d %H:%M')
            disp_table.columns = ['Timestamp', 'Query', 'Relevance Score']

            st.dataframe(
                disp_table,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Query":           st.column_config.TextColumn(width="large"),
                    "Relevance Score": st.column_config.NumberColumn(format="%.3f"),
                },
            )

            csv_unanswered = disp_table.to_csv(index=False)
            st.download_button(
                label="📥 Download Unanswered Queries CSV",
                data=csv_unanswered,
                file_name=f"unanswered_queries_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv",
            )

            st.markdown("---")
            st.subheader("📊 Unanswered Queries Timeline")
            unanswered_df['date'] = unanswered_df['timestamp'].dt.date
            tl = unanswered_df.groupby('date').size().reset_index(name='count')
            fig_tl = px.bar(
                tl, x='date', y='count',
                title='Unanswered Queries per Day',
                color_discrete_sequence=['#e74c3c'],
            )
            st.plotly_chart(fig_tl, use_container_width=True)


# ═════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ═════════════════════════════════════════════════════════════════════════════

def main():
    db_path = "query_analytics.db"
    try:
        create_dashboard(db_path)
    except Exception as e:
        st.error(f"Dashboard error: {e}")
        st.exception(e)


if __name__ == "__main__":
    main()
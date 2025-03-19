import matplotlib.pyplot as plt
from tempo_embeddings.embeddings.weaviate_database import WeaviateDatabaseManager
from tempo_embeddings.visualization.jscatter import JScatterContainer
import pandas as pd
from tqdm import tqdm
from tempo_embeddings.text.corpus import Corpus
from tempo_embeddings.text.passage import Passage
from tempo_embeddings.text.year_span import YearSpan
from tempo_embeddings.settings import STOPWORDS
from tempo_embeddings.text.keyword_extractor import KeywordExtractor


def count_passages(db: WeaviateDatabaseManager, collections: list[str], search_terms: list[str]):
    if not search_terms:
        raise RuntimeError("No seach terms provided.")

    fig, axs = plt.subplots(
        len(collections),
        sharey=True,
        figsize=(3 * len(search_terms), 4 * len(collections)),
    )
    plt.subplots_adjust(hspace=0.5)  # Increase the height space between subplots

    if len(collections) == 1:
        axs = [axs]


    for ax, collection in zip(axs, collections):
        total_count = db.doc_frequency("", collection)

        ax.set_title(f"Matching Passages in '{collection}'")
        ax.set_ylabel("Matching Passages")

        term_freqs = [db.doc_frequency(term, collection) for term in search_terms]

        ax.bar(search_terms, term_freqs)

def construct_frequency_dataframe(db: WeaviateDatabaseManager, search_terms: list[str], collections: list[str], year_range: tuple[int, int]):
    doc_freqs = pd.DataFrame(
        columns=search_terms,
        index=pd.MultiIndex.from_product(
            (
                collections,
                range(int(year_range[0]), int(year_range[1])),
            ),
            names=["collection", "year"],
        ),
    )

    for collection in tqdm(collections, unit="collection"):
        for term in search_terms:
            term_freqs = db.doc_frequencies_per_year(
                term, collection, int(year_range[0]), int(year_range[1])
            )
            for year, freq in term_freqs.items():
                doc_freqs.at[(collection, year), term] = freq
    
    return doc_freqs

def visualise_freq_dataframe(doc_freqs: pd.DataFrame, window_size: int = 1):
    for collection in doc_freqs.index.get_level_values("collection").unique():
        doc_freqs.loc[collection].rolling(window_size).mean().plot(
            figsize=(20, 5),
            title=f"Frequency in '{collection} (Rolling Average over {window_size} years)",
            xlabel="Year",
            ylabel="Frequency",
        )

def initialize_corpora_from_collections(db: WeaviateDatabaseManager,
                        collections: list[str],
                        search_terms: list[str],
                        year_range: tuple[int, int],
                        max_original_documents: int = 5000):
    corpora = []
    for collection in collections:
        for term in search_terms:
            try:
                corpora.append(
                    db.get_corpus(
                        collection,
                        [term],
                        year_from=year_range[0],
                        year_to=year_range[1],
                        include_embeddings=True,
                        limit=max_original_documents,
                    )
                )
            except RuntimeError as e:
                print(
                    f"Failed to retrieve data for term '{term}', collection '{collection}: {e}"
                )
    return corpora

def expand_corpora_with_neighbours(corpora: list[Corpus], db: WeaviateDatabaseManager, collections: list[str], max_neighbours: int, distance_threshold: float, year_range: tuple[int, int]):
    all_passages: set[Passage] = {
        passage for corpus in corpora for passage in corpus.passages
    }

    neighbours: dict[Corpus, Corpus] = {}
    for collection in tqdm(corpora, unit="collection", desc="Getting Neighbours"):
        try:
            neighbours[collection] = db.neighbours(
                collection,
                k=max_neighbours,
                collections=collections,
                distance=distance_threshold,
                year_span=YearSpan(year_range[0], year_range[1]),
                exclude_passages=all_passages,
            )
        except RuntimeError as e:
            print(f"Error while retrieving collection '{collection}': {e}")

    # Remove empty neighbours collections
    neighbours = {
        collection: _neighbours
        for collection, _neighbours in neighbours.items()
        if len(_neighbours) > 0
    }

    # provide a summary of the corpora and how many neighbours were found
    try:
        label_length: int = max(len(collection.label) for collection in corpora)
    except ValueError as e:
        raise RuntimeError("No corpora have been loaded.") from e

    print(
        f"\n{'Collection Label'.ljust(label_length)}\tSize\tNeighbours with Distance < {distance_threshold}"
    )
    for collection in corpora:
        print(
            f"{collection.label.ljust(label_length)}\t{len(collection)}\t{len(neighbours[collection])}"
        )

    # then, merge the original corpora with the neighbours
    subcorpora = corpora + (list(neighbours.values()) if neighbours else [])

    merged_corpus = sum(subcorpora, Corpus())
    merged_corpus.label = "Initial Corpora plus Neighbours"
    return subcorpora, merged_corpus

def compress_embeddings(corpus: Corpus):
    try:
        corpus.compress_embeddings()  # Compute a 2D representation of the embeddings, changes the corpus object
    except ValueError as e:
        print(
            "Could not compute 2D representation of the embeddings, probably because the corpus is empty."
        )
        raise e

def set_stopwords(corpus: Corpus, search_terms: list[str], stopwords_list: list[str]):
    # Generate a `KeywordExtractor` instance to extract keywords from the corpus
    stopwords_set: set[str] = (
        STOPWORDS
        | set(search_terms)
        | {word.strip() for word in stopwords_list if word.strip()}
    )
    keyword_extractor = KeywordExtractor(corpus, exclude_words=stopwords_set).fit()
    return stopwords_set, keyword_extractor

def visualise_embeddings(subcorpora: list[Corpus], filters: list[str], keyword_extractor: KeywordExtractor):
    visualizer = JScatterContainer(
        subcorpora,
        keyword_extractor=keyword_extractor,
        categorical_fields=filters,
    )
    visualizer.visualize()
    

def compile_new_visualiser(db: WeaviateDatabaseManager,
                        collections: list[str],
                        search_terms: list[str], 
                        year_range: tuple[int, int],
                        distance_threshold: float,
                        max_original_documents: int, 
                        max_neighbours: int,
                        stopwords_list: list[str],
                        count_frequencies: bool = True):
    """
    This method conveniently chains the notebook functionalities to compile a new visualiser.
    """
    if count_frequencies:
        count_passages(db, collections, search_terms)
        doc_freqs = construct_frequency_dataframe(db, search_terms, collections, year_range)
        visualise_freq_dataframe(doc_freqs)
    corpora = initialize_corpora_from_collections(db, collections, search_terms, year_range, max_original_documents)
    subcorpora, corpora_with_neighbours = expand_corpora_with_neighbours(corpora, db, collections, max_neighbours, distance_threshold, year_range)
    compress_embeddings(corpora_with_neighbours)
    stopwords, keyword_extractor = set_stopwords(corpora_with_neighbours, search_terms, stopwords_list)
    filters = ["label"] + list(corpora_with_neighbours.metadata_fields())
    visualise_embeddings(subcorpora, filters, keyword_extractor)


    




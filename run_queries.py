import logging
from engine import RecommendationEngine


def main():
    # Reduce logging noise so we can clearly see top-5 results
    logging.getLogger().setLevel(logging.WARNING)
    logging.getLogger("engine").setLevel(logging.WARNING)

    engine = RecommendationEngine("data/shl_catalog_final.csv")

    queries = [
        "Python developer",
        "SQL database",
        "cognitive",
        "personality",
    ]

    for q in queries:
        print("\n" + "=" * 60)
        print("QUERY:", q)
        print("=" * 60)
        recs = engine.get_balanced_recommendations(q, top_k=5)
        for i, r in enumerate(recs, 1):
            print(f"{i}. {r['name']} ({r.get('test_type', '?')})")


if __name__ == "__main__":
    main()


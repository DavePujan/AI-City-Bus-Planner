
def calculate_metrics(df, routes, allocation):
    """
    More realistic service metrics.
    """

    total_stops = len(df)
    total_routes = len(routes)
    buses_used = sum(v["buses_assigned"] for v in allocation.values())

    # coverage: percent stops served by routes
    served_clusters = set(routes.keys())
    served_stops = df[df["cluster"].isin(served_clusters)]
    coverage = (len(served_stops) / max(1, total_stops)) * 100

    avg_headway = sum(
        v["estimated_headway_hours"]
        for v in allocation.values()
    ) / max(1, len(allocation))

    return {
        "total_stops": total_stops,
        "total_routes": total_routes,
        "buses_used": buses_used,
        "coverage_score": round(coverage, 2),
        "avg_headway_hours": round(avg_headway, 2),
    }

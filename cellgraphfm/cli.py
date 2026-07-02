"""Command-line interface for CellGraphFM."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from cellgraphfm import __version__


def _cmd_info(_args: argparse.Namespace) -> int:
    from cellgraphfm.features import describe_foundation_model, list_foundation_models
    from cellgraphfm.models import list_architectures

    print("CellGraphFM", __version__)
    print("\nGNN architectures:")
    for arch in list_architectures():
        print(f"  - {arch}")

    print("\nFoundation-model interfaces (weights not bundled):")
    for name in list_foundation_models():
        spec = describe_foundation_model(name)
        print(f"  - {spec.name:14s} dim={spec.embed_dim:<5d} {spec.architecture}")
    return 0


def _cmd_demo(args: argparse.Namespace) -> int:
    from cellgraphfm.explain import neighborhood_enrichment, top_interactions
    from cellgraphfm.pipeline import demo_graph

    kwargs: dict[str, object] = {}
    if args.method == "knn":
        kwargs["k"] = args.k
    elif args.method == "radius":
        kwargs["radius"] = args.radius
    elif args.method == "contact":
        kwargs["cell_radii"] = args.radius

    graph = demo_graph(n_cells=args.cells, method=args.method, seed=args.seed, **kwargs)
    print(graph)
    print(f"mean node degree : {graph.degree().mean():.2f}")

    print("\nTop cell-type interactions:")
    for type_a, type_b, count in top_interactions(graph, k=5):
        print(f"  type {type_a} <-> type {type_b}: {count} edges")

    zscore, type_ids = neighborhood_enrichment(
        graph, n_permutations=args.permutations, seed=args.seed
    )
    print("\nSame-type neighbourhood enrichment (z-score):")
    for i, t in enumerate(type_ids.tolist()):
        print(f"  type {t}: {zscore[i, i]:+.2f}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cellgraphfm",
        description="Cellular graph intelligence for computational pathology.",
    )
    parser.add_argument("--version", action="version", version=f"cellgraphfm {__version__}")
    sub = parser.add_subparsers(dest="command")

    info = sub.add_parser("info", help="list architectures and foundation models")
    info.set_defaults(func=_cmd_info)

    demo = sub.add_parser("demo", help="build and summarise a synthetic tissue graph")
    demo.add_argument("--cells", type=int, default=200, help="number of cells")
    demo.add_argument(
        "--method",
        choices=["knn", "radius", "delaunay", "contact"],
        default="knn",
        help="graph construction method",
    )
    demo.add_argument("--k", type=int, default=6, help="neighbours for knn")
    demo.add_argument("--radius", type=float, default=60.0, help="radius / cell radius")
    demo.add_argument("--permutations", type=int, default=100, help="enrichment permutations")
    demo.add_argument("--seed", type=int, default=0, help="random seed")
    demo.set_defaults(func=_cmd_demo)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    func = getattr(args, "func", None)
    if func is None:
        parser.print_help()
        return 0
    return func(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

"""
Decision files from SEG-Y files (one per shot, or one file grouped by source position)
or from a list of traces.

Command line of :mod:`makutu.acquisition.decision_create`.

Usage
-----
    python apps/acquisition/decision_create.py --from-segy shot_1.sgy shot_2.sgy --out survey
    python apps/acquisition/decision_create.py traces.txt --out survey --segy shot_1.sgy shot_2.sgy
"""

import argparse

from makutu.acquisition.decision_create import create_decision_files, read_traces_file, shots_from_segy, shots_from_segy_list, shots_from_traces


def _build_parser():
    p = argparse.ArgumentParser(
        description=__doc__.strip().splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument(
        "traces", nargs="?",
        help="Traces file: first line = word numbers, subsequent lines = "
             "one trace per row.",
    )
    src.add_argument(
        "--from-segy", nargs="+", metavar="FILE",
        help="Read geometry from SEG-Y trace headers.  Pass one file to "
             "group traces by source position into shots; pass multiple "
             "files (one per shot) for one-shot-per-file mode.",
    )

    p.add_argument(
        "--out", required=True,
        help="Output rootname (e.g. 'survey' → survey.mapped.data, etc.).",
    )
    p.add_argument(
        "--segy", nargs="+", metavar="FILE",
        help="SEG-Y files, one per shot (in source-order).  Used as the "
             "file list in the mapped .list file.  When --from-segy is "
             "given, defaults to that same file.",
    )

    # Decision-file word numbers for output
    p.add_argument("--word-sx", type=int, default=22)
    p.add_argument("--word-sy", type=int, default=23)
    p.add_argument("--word-sz", type=int, default=14)
    p.add_argument("--word-rx", type=int, default=24)
    p.add_argument("--word-ry", type=int, default=25)
    p.add_argument("--word-rz", type=int, default=13)
    p.add_argument(
        "--index-words", type=lambda s: list(map(int, s.split(","))),
        default=None,
        metavar="W1,W2,...",
        help="Comma-separated word numbers for the index file.  "
             "Default: source X/Y/Z words.",
    )

    # SEG-Y header word positions (only used with --from-segy)
    p.add_argument("--segy-word-sx", type=int, default=19,
                   help="SEG-Y header word for source X  (default: 19).")
    p.add_argument("--segy-word-sy", type=int, default=20,
                   help="SEG-Y header word for source Y  (default: 20).")
    p.add_argument("--segy-word-sz", type=int, default=13,
                   help="SEG-Y header word for the source depth (default: 13, sdepth).")
    p.add_argument("--segy-word-rx", type=int, default=21,
                   help="SEG-Y header word for receiver X (default: 21).")
    p.add_argument("--segy-word-ry", type=int, default=22,
                   help="SEG-Y header word for receiver Y (default: 22).")
    p.add_argument("--segy-word-rz", type=int, default=11,
                   help="SEG-Y header word for the receiver depth (default: 11, gelev as written by makutu).")
    return p


def main(argv=None):
    args = _build_parser().parse_args(argv)

    src_words = (args.word_sx, args.word_sy, args.word_sz)
    rcv_words = (args.word_rx, args.word_ry, args.word_rz)

    if args.from_segy:
        segy_src = (args.segy_word_sx, args.segy_word_sy, args.segy_word_sz)
        segy_rcv = (args.segy_word_rx, args.segy_word_ry, args.segy_word_rz)
        if len(args.from_segy) == 1:
            shots, auto_segy = shots_from_segy(
                args.from_segy[0],
                segy_src_words=segy_src,
                segy_rcv_words=segy_rcv,
            )
        else:
            shots, auto_segy = shots_from_segy_list(
                args.from_segy,
                segy_src_words=segy_src,
                segy_rcv_words=segy_rcv,
            )
        segy_files = args.segy or auto_segy
    else:
        word_numbers, rows = read_traces_file(args.traces)
        shots = shots_from_traces(rows, word_numbers, src_words, rcv_words)
        segy_files = args.segy

    create_decision_files(
        shots,
        out=args.out,
        word_sx=args.word_sx,
        word_sy=args.word_sy,
        word_sz=args.word_sz,
        word_rx=args.word_rx,
        word_ry=args.word_ry,
        word_rz=args.word_rz,
        index_words=args.index_words,
        segy_files=segy_files,
    )


if __name__ == "__main__":
    main()

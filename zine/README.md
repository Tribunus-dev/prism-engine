# Prism Engine Field Guide

This directory contains a sixteen-page, half-letter portrait zine about Prism
Engine. It is designed for home printing as a saddle-stitched booklet on four
duplex US Letter sheets.

Open `index.html` to read the pages in normal order. Open `imposed.html` to print
the booklet. In the print dialog, choose US Letter, landscape orientation,
two-sided printing, flip on the short edge, actual size or 100% scale, no margins,
and background graphics. The imposed edition contains eight printed sides in
booklet order. After printing, fold each sheet vertically, nest Sheet 4 inside
Sheet 3 inside Sheet 2 inside Sheet 1, and staple along the fold.

The physical page size is 5.5 by 8.5 inches. The imposed sheet size is 11 by 8.5
inches. Do not use “fit to page,” because printer scaling can move text into the
trim and fold safety zones.

Content is defined once in `pages.js` and rendered into both editions. The
reading layout lives in `styles.css`; `print.css` produces individual half-letter
pages; and `imposed.css` produces the four-sheet duplex layout. Architectural
figures are self-contained SVG files under `assets/` so lines and labels remain
sharp in print.

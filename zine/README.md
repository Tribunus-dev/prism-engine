# Tribunus Research Zine — Print-First Field Note

A self-contained, print-first mini zine for Tribunus Research that also works as a polished local website when opened in a browser.

## How to open locally

Simply open the `index.html` file in any modern web browser (Google Chrome or Chromium-based browsers recommended).
You can do this by dragging and dropping `index.html` into a browser window, or right-clicking the file and selecting "Open With > Google Chrome".
No network access, build steps, or local servers are required (`file://` protocol works fine).

## Printing the booklet

To produce the imposed duplex booklet, follow these exact instructions:

1. Open `imposed.html` in Google Chrome.
2. Press Command-P (or Ctrl-P).
3. Select your printer or Save to PDF.
4. Set paper size to US Letter.
5. Set orientation to Landscape.
6. Set scale to 100%.
7. Set margins to None.
8. Enable Background graphics.
9. Enable two-sided printing.
10. Choose Flip on short edge.
11. Print all four sides.
12. Fold every sheet in half vertically.
13. Stack Sheet 2 inside Sheet 1.
14. Staple along the center fold if desired.

### Troubleshooting note

- If page backs print upside down, the printer is using long-edge flipping. Reprint with short-edge flipping.
- If the content is clipped, confirm that browser scale is 100%, margins are set to None, and the printer driver is not applying an additional “fit to page” transformation.
- If QR codes become muddy, enable background graphics, use normal or high print quality, and avoid toner-saving mode.

## Customization

### Editing Content
Content is structured as semantic HTML within `index.html`.
Pages are grouped logically inside `<section class="page">` elements.
Modify the text inside to update claims or details. Note that claims must be honest and evidence-backed per Tribunus principles.
If updating content, remember to also run the imposition script or manually update the content in `imposed.html`.

### Replacing QR URLs
QR codes are contained within the `assets/` directory as inline/static SVGs.
If you need to update a QR code to point to a new URL, generate a new SVG without any external API calls, and replace the corresponding file in `assets/`. Ensure the new SVG is crisp and scales properly.

## File Manifest

- `index.html` - The main readable zine file (screen and print standard).
- `imposed.html` - The pre-imposed layout for manual duplex printing.
- `styles.css` - Screen presentation styles (custom properties, grid layout, shadows, responsive adjustments).
- `print.css` - Print presentation overrides (page geometry, page breaks, removal of screen-only UI elements).
- `assets/` - SVG graphics, architecture diagram, and QR codes.

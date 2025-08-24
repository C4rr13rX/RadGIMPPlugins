# RadGIMPPlugins

Totally rad plugins for GIMP.

1. (GIMP 3.0) Rebuild Pixel Grid - Turn any image into prestine pixel art. Scales pixels up to virtual pixels made out of a collection of real pixels.

Name in menu
Filters > Pixel Art > Rebuild Grid

What this is
Rebuilds blown-up pixel art onto a clean virtual grid while preserving original colors. Uses straight-alpha I/O, modal (palette-preserving) sampling inside tile cores, adaptive gridline detection, conservative anti-halo cleanup, and a fast multithreaded preview. Default virtual pixel size k = 23.

Requirements
• GIMP 3.0 or newer (GIMP 2.x is not supported by this plug-in).
• Windows 10/11, macOS, or Linux. Note: GIMP 3 is not supported on Windows 7; if you are on Win7, upgrade the OS or GIMP.
• No external Python packages required (uses Python 3 via GObject Introspection included with GIMP 3).

Install (all platforms)

1. In GIMP, open: Edit > Preferences > Folders > Plug-Ins. This shows the exact per-user plug-ins directory for your install.
2. Place the file there, either directly or in its own folder. Recommended layout:
   plug-ins/
   pixel\_grid\_rebuilder/
   pixel\_grid\_rebuilder.py
3. macOS/Linux only: make it executable:
   chmod +x pixel\_grid\_rebuilder.py
4. Restart GIMP. You should see it under:
   Filters > Pixel Art > Rebuild Grid

Typical plug-ins locations (examples; always prefer the path shown in Preferences)
• Windows 10/11: %APPDATA%\GIMP\3.0\plug-ins
• macOS: \~/Library/Application Support/GIMP/3.0/plug-ins/
• Linux (native): \~/.config/GIMP/3.0/plug-ins/
• Linux (Flatpak): \~/.var/app/org.gimp.GIMP/config/GIMP/3.0/plug-ins/

Quick start

1. Open an image and select the layer you want to rebuild.
2. Run: Filters > Pixel Art > Rebuild Grid.
3. The preview renders asynchronously. Controls:

   * k (px): virtual pixel size (defaults to 23; adjust with ±, ÷2, ×2, or enter a value).
   * Cohesion (%): tightens tile core sampling to reduce halos.
   * Cleanup (%): conservative edge snapping between adjacent tiles.
   * Finalize to uniform squares: paints a perfectly uniform grid (optional).
   * Expand canvas on Apply: resizes canvas to fit uniformized grid (optional).
4. Click Apply. A new layer named “Virtual Pixels” is created; your original layer is left hidden.

Uninstall
Delete pixel\_grid\_rebuilder.py (or its folder) from the plug-ins directory shown in Preferences > Folders > Plug-Ins, then restart GIMP.

Troubleshooting
• Plug-in not in the menu: confirm the exact plug-ins path in Preferences; on macOS/Linux ensure the file is executable. Restart GIMP after changes.
• Flatpak build: use the Flatpak-scoped path above.
• Windows 7: GIMP 3 is not supported; this plug-in targets GIMP 3 only.
• Preview stalls or UI looks frozen: the plug-in renders in background threads and shows a progress bar; if you closed the dialog mid-render, re-open and try again.


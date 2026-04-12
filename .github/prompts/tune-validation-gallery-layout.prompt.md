---
name: Tune Validation Gallery Layout
description: "Adjust a notebook or script figure layout for validation galleries, especially tighter spacing, larger 3D panels, and clearer spatial structure. Use when refining matplotlib gallery composition in this repository."
argument-hint: "Describe the layout change, for example: move columns closer, enlarge panels e/f, add mesh if useful"
agent: agent
---
Refine the active validation-gallery plotting code in this repository.

Task focus:
- Treat the user's request as a concrete figure-layout adjustment for an existing notebook or plotting script.
- Prioritize layout changes over unrelated refactors.
- Keep scientific meaning, paths, and data-processing logic unchanged unless a layout fix requires a minimal supporting change.

When working on the figure:
- Inspect the current subplot/grid configuration before editing.
- Reduce excessive horizontal or vertical spacing if columns or panels are too far apart.
- Increase the visual prominence of panels `(e)` and `(f)` when the request asks for larger 3D views.
- If the figure contains 3D panels, consider adding or improving a mesh, grid, pane styling, or similar depth cue only when it improves readability without clutter.
- Preserve existing labels, panel naming, legend intent, and manuscript-style presentation unless the user asks otherwise.

Editing rules:
- Make the smallest set of code changes that satisfies the requested layout outcome.
- Prefer adjusting `GridSpec`, subplot spans, figure size, aspect settings, camera/view settings, and 3D axis styling before rewriting plotting logic.
- Do not change hard-coded dataset paths just to make the figure prettier.
- If the target is a notebook, edit the relevant code cell directly and keep the notebook runnable.

Validation:
- Run the affected notebook cell or script if practical.
- Confirm that the updated layout addresses the requested spacing and sizing change.
- Report any constraint that prevents verifying the rendered output.

Response style:
- Apply the change directly.
- Summarize what was adjusted in the layout.
- Mention whether panels `(e)` and `(f)` were enlarged and whether a mesh/grid cue was added.
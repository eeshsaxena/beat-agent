# Part 3 chat: changes to a recreation

Mode: offline grammar (written with these commands in view: shows the fallback works, not that it generalises). Clip: P2 (drums + bass).

**7/7 turns leave the score exactly as expected.**

| Message | Expected | Result | Echo (written by code) |
|---|---|---|---|
| Mute the drums. | drums muted, nothing else changed | OK | MUTE Drums. |
| Make the bass a bit brighter. | bass filter cutoff raised, nothing else changed | OK | BRIGHTNESS Bass: filter cutoff 719 -> 1078 Hz. |
| Play it at 140. | tempo 140 BPM, notes untouched | OK | TEMPO 110.00 -> 140.00 BPM (loop 8.727 -> 6.857 s; pitch unchanged). |
| Turn the bass down a little. | bass level lowered by a small step (1-6 dB) | OK | LEVEL Bass: +0.0 -> -3.0 dB. |
| Add a guitar. | refused: the score has no guitar and notes are never invented; score unchanged | OK | (no change) |
| Undo. | the bass level change is undone | OK | Undid the last change to the recreation. |
| Solo the bass. | bass soloed, nothing else changed | OK | SOLO Bass. |

Agent replies:

* **Mute the drums.** Offline grammar: set mute on Drums.
* **Make the bass a bit brighter.** Offline grammar: brightness on Bass.
* **Play it at 140.** Offline grammar: tempo 140 BPM.
* **Turn the bass down a little.** Offline grammar: set gain on Bass.
* **Add a guitar.** Offline grammar: the recreation can only change the tracks that were transcribed; adding an instrument would mean inventing notes. Tracks: drums (Drums), bass (Bass).
* **Undo.** Offline grammar: undo.
* **Solo the bass.** Offline grammar: set solo on Bass.

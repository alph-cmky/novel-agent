# API Rules

Pre-production: no production clients requiring backward compatibility. Do not keep old request fields, deprecated endpoints, or aliases without a current caller.

SSE handlers must always account for: client disconnect, exception cleanup, task cancellation, session cleanup, and resume after interrupt.

Routes stay thin: `request → service/graph → response`.

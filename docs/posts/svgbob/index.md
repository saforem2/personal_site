# 🫥 svgbob
Sam Foreman
2024-11-15

Playing with [ivanceras/`svgbob`](https://github.com/ivanceras/svgbob)
as an alternative to Mermaid

<svg xmlns="http://www.w3.org/2000/svg" width="456" height="304" class="svgbob">
  <style>.svgbob line, .svgbob path, .svgbob circle, .svgbob rect, .svgbob polygon {
  stroke: #838383;
  stroke-width: 2;
  stroke-opacity: 1;
  fill-opacity: 1;
  stroke-linecap: round;
  stroke-linejoin: miter;
}

.svgbob text {
  white-space: pre;
  fill: #838383;
  font-family: Iosevka Fixed, monospace;
  font-size: 14px;
}

.svgbob rect.backdrop {
  stroke: none;
  fill: rgba(0,0,0,0.0);
}

.svgbob .broken {
  stroke-dasharray: 8;
}

.svgbob .filled {
  fill: #838383;
}

.svgbob .bg_filled {
  fill: rgba(0,0,0,0.0);
  stroke-width: 1;
}

.svgbob .nofill {
  fill: rgba(0,0,0,0.0);
}

.svgbob .end_marked_arrow {
  marker-end: url(#arrow);
}

.svgbob .start_marked_arrow {
  marker-start: url(#arrow);
}

.svgbob .end_marked_diamond {
  marker-end: url(#diamond);
}

.svgbob .start_marked_diamond {
  marker-start: url(#diamond);
}

.svgbob .end_marked_circle {
  marker-end: url(#circle);
}

.svgbob .start_marked_circle {
  marker-start: url(#circle);
}

.svgbob .end_marked_open_circle {
  marker-end: url(#open_circle);
}

.svgbob .start_marked_open_circle {
  marker-start: url(#open_circle);
}

.svgbob .end_marked_big_open_circle {
  marker-end: url(#big_open_circle);
}

.svgbob .start_marked_big_open_circle {
  marker-start: url(#big_open_circle);
}

</style>
  <defs>
    <marker id="arrow" viewBox="-2 -2 8 8" refX="4" refY="2" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
      <polygon points="0,0 0,4 4,2 0,0"></polygon>
    </marker>
    <marker id="diamond" viewBox="-2 -2 8 8" refX="4" refY="2" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
      <polygon points="0,2 2,0 4,2 2,4 0,2"></polygon>
    </marker>
    <marker id="circle" viewBox="0 0 8 8" refX="4" refY="4" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
      <circle cx="4" cy="4" r="2" class="filled"></circle>
    </marker>
    <marker id="open_circle" viewBox="0 0 8 8" refX="4" refY="4" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
      <circle cx="4" cy="4" r="2" class="bg_filled"></circle>
    </marker>
    <marker id="big_open_circle" viewBox="0 0 8 8" refX="4" refY="4" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
      <circle cx="4" cy="4" r="3" class="bg_filled"></circle>
    </marker>
  </defs>
  <rect class="backdrop" x="0" y="0" width="456" height="304"></rect>
  <rect x="156" y="40" width="80" height="32" class="solid nofill" rx="4"></rect>
  <text x="170" y="60" >Network</text>
  <rect x="332" y="40" width="88" height="32" class="solid nofill" rx="4"></rect>
  <text x="346" y="60" >Loss</text>
  <path d="M 384,48 A 16,16 0,0,0 384,64" class="nofill"></path>
  <text x="386" y="60" >x0</text>
  <path d="M 400,48 A 16,16 0,0,1 400,64" class="nofill"></path>
  <rect x="156" y="136" width="80" height="32" class="solid nofill" rx="4"></rect>
  <text x="170" y="156" >Network</text>
  <rect x="332" y="136" width="88" height="32" class="solid nofill" rx="4"></rect>
  <text x="346" y="156" >Loss</text>
  <path d="M 384,144 A 16,16 0,0,0 384,160" class="nofill"></path>
  <text x="386" y="156" >x1</text>
  <path d="M 400,144 A 16,16 0,0,1 400,160" class="nofill"></path>
  <rect x="156" y="232" width="80" height="32" class="solid nofill" rx="4"></rect>
  <text x="170" y="252" >Network</text>
  <rect x="332" y="232" width="88" height="32" class="solid nofill" rx="4"></rect>
  <text x="346" y="252" >Loss</text>
  <path d="M 384,240 A 16,16 0,0,0 384,256" class="nofill"></path>
  <text x="386" y="252" >x2</text>
  <path d="M 400,240 A 16,16 0,0,1 400,256" class="nofill"></path>
  <text x="34" y="28" >Data</text>
  <text x="138" y="28" >GPU0</text>
  <text x="42" y="60" >x0</text>
  <polygon points="136,52 144,56 136,60" class="filled"></polygon>
  <line x1="92" y1="64" x2="92" y2="144" class="solid"></line>
  <text x="138" y="124" >GPU1</text>
  <polygon points="296,132 288,136 296,140" class="filled"></polygon>
  <text x="42" y="156" >x1</text>
  <polygon points="136,148 144,152 136,156" class="filled"></polygon>
  <line x1="92" y1="160" x2="92" y2="240" class="solid"></line>
  <polygon points="272,164 280,168 272,172" class="filled"></polygon>
  <text x="138" y="220" >GPU2</text>
  <polygon points="296,228 288,232 296,236" class="filled"></polygon>
  <text x="42" y="252" >x2</text>
  <polygon points="136,244 144,248 136,252" class="filled"></polygon>
  <polygon points="272,260 280,264 272,268" class="filled"></polygon>
  <polygon points="256,36 248,40 256,44" class="filled"></polygon>
  <line x1="256" y1="40" x2="308" y2="40" class="solid end_marked_open_circle"></line>
  <line x1="264" y1="56" x2="252" y2="56" class="solid end_marked_open_circle"></line>
  <line x1="264" y1="56" x2="312" y2="56" class="solid"></line>
  <polygon points="312,52 320,56 312,60" class="filled"></polygon>
  <g>
    <path d="M 8,8 A 4,4 0,0,0 4,12" class="nofill"></path>
    <line x1="4" y1="12" x2="4" y2="276" class="solid"></line>
    <line x1="8" y1="8" x2="88" y2="8" class="solid"></line>
    <path d="M 88,8 A 4,4 0,0,1 92,12" class="nofill"></path>
    <line x1="92" y1="12" x2="92" y2="48" class="solid"></line>
    <path d="M 4,276 A 4,4 0,0,0 8,280" class="nofill"></path>
    <line x1="8" y1="280" x2="88" y2="280" class="solid"></line>
    <line x1="92" y1="256" x2="92" y2="276" class="solid"></line>
    <path d="M 92,276 A 4,4 0,0,1 88,280" class="nofill"></path>
  </g>
  <g>
    <path d="M 128,8 A 4,4 0,0,0 124,12" class="nofill"></path>
    <line x1="124" y1="12" x2="124" y2="48" class="solid"></line>
    <line x1="128" y1="8" x2="440" y2="8" class="solid"></line>
    <path d="M 440,8 A 4,4 0,0,1 444,12" class="nofill"></path>
    <line x1="444" y1="12" x2="444" y2="84" class="solid"></line>
    <line x1="124" y1="64" x2="124" y2="84" class="solid"></line>
    <path d="M 124,84 A 4,4 0,0,0 128,88" class="nofill"></path>
    <line x1="128" y1="88" x2="440" y2="88" class="solid"></line>
    <path d="M 444,84 A 4,4 0,0,1 440,88" class="nofill"></path>
  </g>
  <g>
    <path d="M 24,40 A 4,4 0,0,0 20,44" class="nofill"></path>
    <line x1="20" y1="44" x2="20" y2="68" class="solid"></line>
    <line x1="24" y1="40" x2="72" y2="40" class="solid"></line>
    <path d="M 72,40 A 4,4 0,0,1 76,44" class="nofill"></path>
    <line x1="76" y1="44" x2="76" y2="68" class="solid"></line>
    <line x1="76" y1="56" x2="136" y2="56" class="solid"></line>
    <path d="M 20,68 A 4,4 0,0,0 24,72" class="nofill"></path>
    <line x1="24" y1="72" x2="72" y2="72" class="solid"></line>
    <path d="M 76,68 A 4,4 0,0,1 72,72" class="nofill"></path>
  </g>
  <g>
    <path d="M 128,104 A 4,4 0,0,0 124,108" class="nofill"></path>
    <line x1="124" y1="108" x2="124" y2="144" class="solid"></line>
    <line x1="128" y1="104" x2="440" y2="104" class="solid"></line>
    <path d="M 440,104 A 4,4 0,0,1 444,108" class="nofill"></path>
    <line x1="444" y1="108" x2="444" y2="180" class="solid"></line>
    <line x1="124" y1="160" x2="124" y2="180" class="solid"></line>
    <path d="M 124,180 A 4,4 0,0,0 128,184" class="nofill"></path>
    <line x1="128" y1="184" x2="440" y2="184" class="solid"></line>
    <path d="M 444,180 A 4,4 0,0,1 440,184" class="nofill"></path>
  </g>
  <g>
    <path d="M 24,136 A 4,4 0,0,0 20,140" class="nofill"></path>
    <line x1="20" y1="140" x2="20" y2="164" class="solid"></line>
    <line x1="24" y1="136" x2="72" y2="136" class="solid"></line>
    <path d="M 72,136 A 4,4 0,0,1 76,140" class="nofill"></path>
    <line x1="76" y1="140" x2="76" y2="164" class="solid"></line>
    <line x1="76" y1="152" x2="136" y2="152" class="solid"></line>
    <path d="M 20,164 A 4,4 0,0,0 24,168" class="nofill"></path>
    <line x1="24" y1="168" x2="72" y2="168" class="solid"></line>
    <path d="M 76,164 A 4,4 0,0,1 72,168" class="nofill"></path>
  </g>
  <g>
    <path d="M 272,136 A 8,8 0,0,0 266,140" class="nofill"></path>
    <line x1="266" y1="140" x2="264" y2="144" class="solid"></line>
    <path d="M 264,144 A 16,16 0,0,0 264,160" class="nofill"></path>
    <line x1="264" y1="160" x2="266" y2="164" class="solid"></line>
    <path d="M 266,164 A 8,8 0,0,0 272,168" class="nofill"></path>
  </g>
  <g>
    <path d="M 296,136 A 8,8 0,0,1 302,140" class="nofill"></path>
    <line x1="302" y1="140" x2="304" y2="144" class="solid"></line>
    <path d="M 304,144 A 16,16 0,0,1 304,160" class="nofill"></path>
    <line x1="304" y1="160" x2="302" y2="164" class="solid"></line>
    <path d="M 302,164 A 8,8 0,0,1 296,168" class="nofill"></path>
  </g>
  <g>
    <path d="M 128,200 A 4,4 0,0,0 124,204" class="nofill"></path>
    <line x1="124" y1="204" x2="124" y2="240" class="solid"></line>
    <line x1="128" y1="200" x2="440" y2="200" class="solid"></line>
    <path d="M 440,200 A 4,4 0,0,1 444,204" class="nofill"></path>
    <line x1="444" y1="204" x2="444" y2="276" class="solid"></line>
    <line x1="124" y1="256" x2="124" y2="276" class="solid"></line>
    <path d="M 124,276 A 4,4 0,0,0 128,280" class="nofill"></path>
    <line x1="128" y1="280" x2="440" y2="280" class="solid"></line>
    <path d="M 444,276 A 4,4 0,0,1 440,280" class="nofill"></path>
  </g>
  <g>
    <path d="M 24,232 A 4,4 0,0,0 20,236" class="nofill"></path>
    <line x1="20" y1="236" x2="20" y2="260" class="solid"></line>
    <line x1="24" y1="232" x2="72" y2="232" class="solid"></line>
    <path d="M 72,232 A 4,4 0,0,1 76,236" class="nofill"></path>
    <line x1="76" y1="236" x2="76" y2="260" class="solid"></line>
    <line x1="76" y1="248" x2="136" y2="248" class="solid"></line>
    <path d="M 20,260 A 4,4 0,0,0 24,264" class="nofill"></path>
    <line x1="24" y1="264" x2="72" y2="264" class="solid"></line>
    <path d="M 76,260 A 4,4 0,0,1 72,264" class="nofill"></path>
  </g>
  <g>
    <path d="M 272,232 A 8,8 0,0,0 266,236" class="nofill"></path>
    <line x1="266" y1="236" x2="264" y2="240" class="solid"></line>
    <path d="M 264,240 A 16,16 0,0,0 264,256" class="nofill"></path>
    <line x1="264" y1="256" x2="266" y2="260" class="solid"></line>
    <path d="M 266,260 A 8,8 0,0,0 272,264" class="nofill"></path>
  </g>
  <g>
    <path d="M 296,232 A 8,8 0,0,1 302,236" class="nofill"></path>
    <line x1="302" y1="236" x2="304" y2="240" class="solid"></line>
    <path d="M 304,240 A 16,16 0,0,1 304,256" class="nofill"></path>
    <line x1="304" y1="256" x2="302" y2="260" class="solid"></line>
    <path d="M 302,260 A 8,8 0,0,1 296,264" class="nofill"></path>
  </g>
</svg>

<svg xmlns="http://www.w3.org/2000/svg" width="480" height="560" class="svgbob">
  <style>.svgbob line, .svgbob path, .svgbob circle, .svgbob rect, .svgbob polygon {
  stroke: #838383;
  stroke-width: 2;
  stroke-opacity: 1;
  fill-opacity: 1;
  stroke-linecap: round;
  stroke-linejoin: miter;
}

.svgbob text {
  white-space: pre;
  fill: #838383;
  font-family: Iosevka Fixed, monospace;
  font-size: 14px;
}

.svgbob rect.backdrop {
  stroke: none;
  fill: rgba(0,0,0,0.0);
}

.svgbob .broken {
  stroke-dasharray: 8;
}

.svgbob .filled {
  fill: #838383;
}

.svgbob .bg_filled {
  fill: rgba(0,0,0,0.0);
  stroke-width: 1;
}

.svgbob .nofill {
  fill: rgba(0,0,0,0.0);
}

.svgbob .end_marked_arrow {
  marker-end: url(#arrow);
}

.svgbob .start_marked_arrow {
  marker-start: url(#arrow);
}

.svgbob .end_marked_diamond {
  marker-end: url(#diamond);
}

.svgbob .start_marked_diamond {
  marker-start: url(#diamond);
}

.svgbob .end_marked_circle {
  marker-end: url(#circle);
}

.svgbob .start_marked_circle {
  marker-start: url(#circle);
}

.svgbob .end_marked_open_circle {
  marker-end: url(#open_circle);
}

.svgbob .start_marked_open_circle {
  marker-start: url(#open_circle);
}

.svgbob .end_marked_big_open_circle {
  marker-end: url(#big_open_circle);
}

.svgbob .start_marked_big_open_circle {
  marker-start: url(#big_open_circle);
}

</style>
  <defs>
    <marker id="arrow" viewBox="-2 -2 8 8" refX="4" refY="2" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
      <polygon points="0,0 0,4 4,2 0,0"></polygon>
    </marker>
    <marker id="diamond" viewBox="-2 -2 8 8" refX="4" refY="2" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
      <polygon points="0,2 2,0 4,2 2,4 0,2"></polygon>
    </marker>
    <marker id="circle" viewBox="0 0 8 8" refX="4" refY="4" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
      <circle cx="4" cy="4" r="2" class="filled"></circle>
    </marker>
    <marker id="open_circle" viewBox="0 0 8 8" refX="4" refY="4" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
      <circle cx="4" cy="4" r="2" class="bg_filled"></circle>
    </marker>
    <marker id="big_open_circle" viewBox="0 0 8 8" refX="4" refY="4" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
      <circle cx="4" cy="4" r="3" class="bg_filled"></circle>
    </marker>
  </defs>
  <rect class="backdrop" x="0" y="0" width="480" height="560"></rect>
  <rect x="220" y="56" width="80" height="32" class="solid nofill" rx="4"></rect>
  <text x="234" y="76" >Network</text>
  <rect x="372" y="56" width="56" height="32" class="solid nofill" rx="4"></rect>
  <text x="386" y="76" >Loss</text>
  <rect x="220" y="248" width="80" height="32" class="solid nofill" rx="4"></rect>
  <text x="234" y="268" >Network</text>
  <rect x="372" y="248" width="56" height="32" class="solid nofill" rx="4"></rect>
  <text x="386" y="268" >Loss</text>
  <rect x="220" y="440" width="80" height="32" class="solid nofill" rx="4"></rect>
  <text x="234" y="460" >Network</text>
  <rect x="372" y="440" width="56" height="32" class="solid nofill" rx="4"></rect>
  <text x="386" y="460" >Loss</text>
  <text x="34" y="28" >Data</text>
  <text x="202" y="28" >GPU0</text>
  <text x="42" y="76" >x0</text>
  <polygon points="200,68 208,72 200,76" class="filled"></polygon>
  <line x1="92" y1="80" x2="92" y2="256" class="solid"></line>
  <polygon points="256,108 260,96 264,108" class="filled"></polygon>
  <text x="202" y="204" >GPU1</text>
  <polygon points="400,228 408,228 404,240" class="filled"></polygon>
  <text x="42" y="268" >x1</text>
  <polygon points="200,260 208,264 200,268" class="filled"></polygon>
  <line x1="92" y1="272" x2="92" y2="448" class="solid"></line>
  <polygon points="256,300 260,288 264,300" class="filled"></polygon>
  <text x="202" y="396" >GPU2</text>
  <polygon points="400,420 408,420 404,432" class="filled"></polygon>
  <text x="42" y="460" >x2</text>
  <polygon points="200,452 208,456 200,460" class="filled"></polygon>
  <polygon points="256,492 260,480 264,492" class="filled"></polygon>
  <line x1="328" y1="72" x2="316" y2="72" class="solid end_marked_open_circle"></line>
  <line x1="328" y1="72" x2="352" y2="72" class="solid"></line>
  <polygon points="352,68 360,72 352,76" class="filled"></polygon>
  <g>
    <path d="M 8,8 A 4,4 0,0,0 4,12" class="nofill"></path>
    <line x1="4" y1="12" x2="4" y2="532" class="solid"></line>
    <line x1="8" y1="8" x2="88" y2="8" class="solid"></line>
    <path d="M 88,8 A 4,4 0,0,1 92,12" class="nofill"></path>
    <line x1="92" y1="12" x2="92" y2="64" class="solid"></line>
    <path d="M 4,532 A 4,4 0,0,0 8,536" class="nofill"></path>
    <line x1="8" y1="536" x2="88" y2="536" class="solid"></line>
    <line x1="92" y1="464" x2="92" y2="532" class="solid"></line>
    <path d="M 92,532 A 4,4 0,0,1 88,536" class="nofill"></path>
  </g>
  <g>
    <path d="M 192,8 A 4,4 0,0,0 188,12" class="nofill"></path>
    <line x1="188" y1="12" x2="188" y2="64" class="solid"></line>
    <line x1="192" y1="8" x2="464" y2="8" class="solid"></line>
    <path d="M 464,8 A 4,4 0,0,1 468,12" class="nofill"></path>
    <line x1="468" y1="12" x2="468" y2="148" class="solid"></line>
    <line x1="188" y1="80" x2="188" y2="148" class="solid"></line>
    <path d="M 188,148 A 4,4 0,0,0 192,152" class="nofill"></path>
    <line x1="192" y1="152" x2="464" y2="152" class="solid"></line>
    <path d="M 468,148 A 4,4 0,0,1 464,152" class="nofill"></path>
  </g>
  <g>
    <path d="M 24,56 A 4,4 0,0,0 20,60" class="nofill"></path>
    <line x1="20" y1="60" x2="20" y2="84" class="solid"></line>
    <line x1="24" y1="56" x2="72" y2="56" class="solid"></line>
    <path d="M 72,56 A 4,4 0,0,1 76,60" class="nofill"></path>
    <line x1="76" y1="60" x2="76" y2="84" class="solid"></line>
    <line x1="76" y1="72" x2="200" y2="72" class="solid"></line>
    <path d="M 20,84 A 4,4 0,0,0 24,88" class="nofill"></path>
    <line x1="24" y1="88" x2="72" y2="88" class="solid"></line>
    <path d="M 76,84 A 4,4 0,0,1 72,88" class="nofill"></path>
  </g>
  <g>
    <line x1="260" y1="108" x2="260" y2="132" class="solid"></line>
    <path d="M 260,132 A 4,4 0,0,0 264,136" class="nofill"></path>
    <line x1="264" y1="136" x2="400" y2="136" class="solid"></line>
    <line x1="404" y1="96" x2="404" y2="132" class="solid"></line>
    <path d="M 404,132 A 4,4 0,0,1 400,136" class="nofill"></path>
  </g>
  <g>
    <path d="M 192,184 A 4,4 0,0,0 188,188" class="nofill"></path>
    <line x1="188" y1="188" x2="188" y2="256" class="solid"></line>
    <line x1="192" y1="184" x2="464" y2="184" class="solid"></line>
    <path d="M 464,184 A 4,4 0,0,1 468,188" class="nofill"></path>
    <line x1="468" y1="188" x2="468" y2="340" class="solid"></line>
    <line x1="188" y1="272" x2="188" y2="340" class="solid"></line>
    <path d="M 188,340 A 4,4 0,0,0 192,344" class="nofill"></path>
    <line x1="192" y1="344" x2="464" y2="344" class="solid"></line>
    <path d="M 468,340 A 4,4 0,0,1 464,344" class="nofill"></path>
  </g>
  <g>
    <path d="M 264,200 A 4,4 0,0,0 260,204" class="nofill"></path>
    <line x1="260" y1="204" x2="260" y2="240" class="solid"></line>
    <line x1="264" y1="200" x2="400" y2="200" class="solid"></line>
    <path d="M 400,200 A 4,4 0,0,1 404,204" class="nofill"></path>
    <line x1="404" y1="204" x2="404" y2="228" class="solid"></line>
  </g>
  <g>
    <path d="M 24,248 A 4,4 0,0,0 20,252" class="nofill"></path>
    <line x1="20" y1="252" x2="20" y2="276" class="solid"></line>
    <line x1="24" y1="248" x2="72" y2="248" class="solid"></line>
    <path d="M 72,248 A 4,4 0,0,1 76,252" class="nofill"></path>
    <line x1="76" y1="252" x2="76" y2="276" class="solid"></line>
    <line x1="76" y1="264" x2="200" y2="264" class="solid"></line>
    <path d="M 20,276 A 4,4 0,0,0 24,280" class="nofill"></path>
    <line x1="24" y1="280" x2="72" y2="280" class="solid"></line>
    <path d="M 76,276 A 4,4 0,0,1 72,280" class="nofill"></path>
  </g>
  <g>
    <line x1="260" y1="300" x2="260" y2="324" class="solid"></line>
    <path d="M 260,324 A 4,4 0,0,0 264,328" class="nofill"></path>
    <line x1="264" y1="328" x2="400" y2="328" class="solid"></line>
    <line x1="404" y1="288" x2="404" y2="324" class="solid"></line>
    <path d="M 404,324 A 4,4 0,0,1 400,328" class="nofill"></path>
  </g>
  <g>
    <path d="M 192,376 A 4,4 0,0,0 188,380" class="nofill"></path>
    <line x1="188" y1="380" x2="188" y2="448" class="solid"></line>
    <line x1="192" y1="376" x2="464" y2="376" class="solid"></line>
    <path d="M 464,376 A 4,4 0,0,1 468,380" class="nofill"></path>
    <line x1="468" y1="380" x2="468" y2="532" class="solid"></line>
    <line x1="188" y1="464" x2="188" y2="532" class="solid"></line>
    <path d="M 188,532 A 4,4 0,0,0 192,536" class="nofill"></path>
    <line x1="192" y1="536" x2="464" y2="536" class="solid"></line>
    <path d="M 468,532 A 4,4 0,0,1 464,536" class="nofill"></path>
  </g>
  <g>
    <path d="M 264,392 A 4,4 0,0,0 260,396" class="nofill"></path>
    <line x1="260" y1="396" x2="260" y2="432" class="solid"></line>
    <line x1="264" y1="392" x2="400" y2="392" class="solid"></line>
    <path d="M 400,392 A 4,4 0,0,1 404,396" class="nofill"></path>
    <line x1="404" y1="396" x2="404" y2="420" class="solid"></line>
  </g>
  <g>
    <path d="M 24,440 A 4,4 0,0,0 20,444" class="nofill"></path>
    <line x1="20" y1="444" x2="20" y2="468" class="solid"></line>
    <line x1="24" y1="440" x2="72" y2="440" class="solid"></line>
    <path d="M 72,440 A 4,4 0,0,1 76,444" class="nofill"></path>
    <line x1="76" y1="444" x2="76" y2="468" class="solid"></line>
    <line x1="76" y1="456" x2="200" y2="456" class="solid"></line>
    <path d="M 20,468 A 4,4 0,0,0 24,472" class="nofill"></path>
    <line x1="24" y1="472" x2="72" y2="472" class="solid"></line>
    <path d="M 76,468 A 4,4 0,0,1 72,472" class="nofill"></path>
  </g>
  <g>
    <line x1="260" y1="492" x2="260" y2="516" class="solid"></line>
    <path d="M 260,516 A 4,4 0,0,0 264,520" class="nofill"></path>
    <line x1="264" y1="520" x2="400" y2="520" class="solid"></line>
    <line x1="404" y1="480" x2="404" y2="516" class="solid"></line>
    <path d="M 404,516 A 4,4 0,0,1 400,520" class="nofill"></path>
  </g>
</svg>

<svg xmlns="http://www.w3.org/2000/svg" width="232" height="192" class="svgbob">
  <style>.svgbob line, .svgbob path, .svgbob circle, .svgbob rect, .svgbob polygon {
  stroke: #838383;
  stroke-width: 2;
  stroke-opacity: 1;
  fill-opacity: 1;
  stroke-linecap: round;
  stroke-linejoin: miter;
}

.svgbob text {
  white-space: pre;
  fill: #838383;
  font-family: Iosevka Fixed, monospace;
  font-size: 14px;
}

.svgbob rect.backdrop {
  stroke: none;
  fill: rgba(0,0,0,0.0);
}

.svgbob .broken {
  stroke-dasharray: 8;
}

.svgbob .filled {
  fill: #838383;
}

.svgbob .bg_filled {
  fill: rgba(0,0,0,0.0);
  stroke-width: 1;
}

.svgbob .nofill {
  fill: rgba(0,0,0,0.0);
}

.svgbob .end_marked_arrow {
  marker-end: url(#arrow);
}

.svgbob .start_marked_arrow {
  marker-start: url(#arrow);
}

.svgbob .end_marked_diamond {
  marker-end: url(#diamond);
}

.svgbob .start_marked_diamond {
  marker-start: url(#diamond);
}

.svgbob .end_marked_circle {
  marker-end: url(#circle);
}

.svgbob .start_marked_circle {
  marker-start: url(#circle);
}

.svgbob .end_marked_open_circle {
  marker-end: url(#open_circle);
}

.svgbob .start_marked_open_circle {
  marker-start: url(#open_circle);
}

.svgbob .end_marked_big_open_circle {
  marker-end: url(#big_open_circle);
}

.svgbob .start_marked_big_open_circle {
  marker-start: url(#big_open_circle);
}

</style>
  <defs>
    <marker id="arrow" viewBox="-2 -2 8 8" refX="4" refY="2" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
      <polygon points="0,0 0,4 4,2 0,0"></polygon>
    </marker>
    <marker id="diamond" viewBox="-2 -2 8 8" refX="4" refY="2" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
      <polygon points="0,2 2,0 4,2 2,4 0,2"></polygon>
    </marker>
    <marker id="circle" viewBox="0 0 8 8" refX="4" refY="4" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
      <circle cx="4" cy="4" r="2" class="filled"></circle>
    </marker>
    <marker id="open_circle" viewBox="0 0 8 8" refX="4" refY="4" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
      <circle cx="4" cy="4" r="2" class="bg_filled"></circle>
    </marker>
    <marker id="big_open_circle" viewBox="0 0 8 8" refX="4" refY="4" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
      <circle cx="4" cy="4" r="3" class="bg_filled"></circle>
    </marker>
  </defs>
  <rect class="backdrop" x="0" y="0" width="232" height="192"></rect>
  <rect x="108" y="56" width="16" height="16" class="solid nofill" rx="4"></rect>
  <rect x="140" y="56" width="16" height="16" class="solid nofill" rx="4"></rect>
  <rect x="172" y="56" width="16" height="16" class="solid nofill" rx="4"></rect>
  <polygon points="176,36 184,36 180,48" class="filled"></polygon>
  <polygon points="96,68 104,72 96,76" class="filled"></polygon>
  <line x1="120" y1="80" x2="130" y2="100" class="solid"></line>
  <line x1="176" y1="80" x2="166" y2="100" class="solid"></line>
  <polygon points="127,100 134,108 134,97" class="filled"></polygon>
  <polygon points="169,100 162,108 162,97" class="filled"></polygon>
  <polygon points="168,164 176,168 168,172" class="filled"></polygon>
  <g>
    <path d="M 184,8 A 4,4 0,0,0 180,12" class="nofill"></path>
    <line x1="180" y1="12" x2="180" y2="36" class="solid"></line>
    <line x1="184" y1="8" x2="200" y2="8" class="solid"></line>
    <path d="M 200,8 A 4,4 0,0,1 204,12" class="nofill"></path>
    <line x1="204" y1="12" x2="204" y2="128" class="solid"></line>
    <path d="M 80,72 A 4,4 0,0,0 76,76" class="nofill"></path>
    <line x1="76" y1="76" x2="76" y2="128" class="solid"></line>
    <line x1="80" y1="72" x2="96" y2="72" class="solid"></line>
    <line x1="148" y1="104" x2="136" y2="128" class="solid"></line>
    <line x1="148" y1="104" x2="160" y2="128" class="solid"></line>
    <line x1="76" y1="128" x2="136" y2="128" class="solid"></line>
    <line x1="160" y1="128" x2="204" y2="128" class="solid"></line>
    <line x1="136" y1="128" x2="148" y2="152" class="solid"></line>
    <line x1="160" y1="128" x2="148" y2="152" class="solid"></line>
    <line x1="148" y1="152" x2="148" y2="164" class="solid"></line>
    <path d="M 148,164 A 4,4 0,0,0 152,168" class="nofill"></path>
    <line x1="152" y1="168" x2="168" y2="168" class="solid"></line>
  </g>
  <g>
    <line x1="148" y1="80" x2="148" y2="100" class="solid"></line>
    <path d="M 148,100 A 16,16 0,0,1 146,108" class="nofill"></path>
    <path d="M 148,100 A 16,16 0,0,0 150,108" class="nofill"></path>
  </g>
  <g>
    <line x1="192" y1="160" x2="224" y2="160" class="solid"></line>
    <line x1="192" y1="160" x2="184" y2="176" class="solid"></line>
    <line x1="184" y1="176" x2="216" y2="176" class="solid"></line>
    <line x1="224" y1="160" x2="216" y2="176" class="solid"></line>
  </g>
</svg>

<svg xmlns="http://www.w3.org/2000/svg" width="432" height="288" class="svgbob">
  <style>.svgbob line, .svgbob path, .svgbob circle, .svgbob rect, .svgbob polygon {
  stroke: #838383;
  stroke-width: 2;
  stroke-opacity: 1;
  fill-opacity: 1;
  stroke-linecap: round;
  stroke-linejoin: miter;
}

.svgbob text {
  white-space: pre;
  fill: #838383;
  font-family: Iosevka Fixed, monospace;
  font-size: 14px;
}

.svgbob rect.backdrop {
  stroke: none;
  fill: rgba(0,0,0,0.0);
}

.svgbob .broken {
  stroke-dasharray: 8;
}

.svgbob .filled {
  fill: #838383;
}

.svgbob .bg_filled {
  fill: rgba(0,0,0,0.0);
  stroke-width: 1;
}

.svgbob .nofill {
  fill: rgba(0,0,0,0.0);
}

.svgbob .end_marked_arrow {
  marker-end: url(#arrow);
}

.svgbob .start_marked_arrow {
  marker-start: url(#arrow);
}

.svgbob .end_marked_diamond {
  marker-end: url(#diamond);
}

.svgbob .start_marked_diamond {
  marker-start: url(#diamond);
}

.svgbob .end_marked_circle {
  marker-end: url(#circle);
}

.svgbob .start_marked_circle {
  marker-start: url(#circle);
}

.svgbob .end_marked_open_circle {
  marker-end: url(#open_circle);
}

.svgbob .start_marked_open_circle {
  marker-start: url(#open_circle);
}

.svgbob .end_marked_big_open_circle {
  marker-end: url(#big_open_circle);
}

.svgbob .start_marked_big_open_circle {
  marker-start: url(#big_open_circle);
}

</style>
  <defs>
    <marker id="arrow" viewBox="-2 -2 8 8" refX="4" refY="2" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
      <polygon points="0,0 0,4 4,2 0,0"></polygon>
    </marker>
    <marker id="diamond" viewBox="-2 -2 8 8" refX="4" refY="2" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
      <polygon points="0,2 2,0 4,2 2,4 0,2"></polygon>
    </marker>
    <marker id="circle" viewBox="0 0 8 8" refX="4" refY="4" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
      <circle cx="4" cy="4" r="2" class="filled"></circle>
    </marker>
    <marker id="open_circle" viewBox="0 0 8 8" refX="4" refY="4" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
      <circle cx="4" cy="4" r="2" class="bg_filled"></circle>
    </marker>
    <marker id="big_open_circle" viewBox="0 0 8 8" refX="4" refY="4" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
      <circle cx="4" cy="4" r="3" class="bg_filled"></circle>
    </marker>
  </defs>
  <rect class="backdrop" x="0" y="0" width="432" height="288"></rect>
  <rect x="84" y="8" width="32" height="16" class="solid nofill" rx="4"></rect>
  <rect x="140" y="8" width="32" height="16" class="solid nofill" rx="4"></rect>
  <rect x="188" y="8" width="32" height="16" class="solid nofill" rx="4"></rect>
  <rect x="244" y="8" width="32" height="16" class="solid nofill" rx="4"></rect>
  <rect x="316" y="8" width="32" height="16" class="solid nofill" rx="4"></rect>
  <rect x="84" y="72" width="104" height="32" class="solid nofill" rx="4"></rect>
  <text x="98" y="92" >Filesystem</text>
  <rect x="220" y="72" width="96" height="32" class="solid nofill" rx="4"></rect>
  <text x="234" y="92" >Scheduler</text>
  <rect x="116" y="152" width="40" height="32" class="solid nofill" rx="4"></rect>
  <text x="130" y="172" >IO</text>
  <rect x="300" y="152" width="80" height="32" class="solid nofill" rx="4"></rect>
  <text x="314" y="172" >Network</text>
  <rect x="100" y="232" width="320" height="32" class="solid nofill" rx="4"></rect>
  <text x="250" y="252" >HAL</text>
  <rect x="372" y="8" width="32" height="16" class="solid nofill" rx="4"></rect>
  <rect x="356" y="72" width="48" height="32" class="solid nofill" rx="4"></rect>
  <text x="370" y="92" >MMU</text>
  <line x1="100" y1="32" x2="100" y2="52" class="solid"></line>
  <polygon points="96,52 104,52 100,64" class="filled"></polygon>
  <line x1="156" y1="32" x2="156" y2="52" class="solid"></line>
  <polygon points="152,52 160,52 156,64" class="filled"></polygon>
  <polygon points="168,164 160,168 168,172" class="filled"></polygon>
  <line x1="260" y1="32" x2="260" y2="52" class="solid"></line>
  <polygon points="256,52 264,52 260,64" class="filled"></polygon>
  <line x1="332" y1="32" x2="332" y2="132" class="solid"></line>
  <polygon points="328,132 336,132 332,144" class="filled"></polygon>
  <line x1="140" y1="112" x2="140" y2="132" class="solid"></line>
  <polygon points="136,132 144,132 140,144" class="filled"></polygon>
  <line x1="260" y1="112" x2="260" y2="212" class="solid"></line>
  <polygon points="256,212 264,212 260,224" class="filled"></polygon>
  <line x1="140" y1="192" x2="140" y2="212" class="solid"></line>
  <polygon points="136,212 144,212 140,224" class="filled"></polygon>
  <line x1="340" y1="192" x2="340" y2="212" class="solid"></line>
  <polygon points="336,212 344,212 340,224" class="filled"></polygon>
  <line x1="388" y1="32" x2="388" y2="52" class="solid"></line>
  <polygon points="384,52 392,52 388,64" class="filled"></polygon>
  <text x="10" y="28" >OS</text>
  <text x="34" y="28" >API</text>
  <g>
    <line x1="204" y1="32" x2="204" y2="164" class="solid"></line>
    <line x1="168" y1="168" x2="200" y2="168" class="solid"></line>
    <path d="M 204,164 A 4,4 0,0,1 200,168" class="nofill"></path>
  </g>
</svg>

<svg xmlns="http://www.w3.org/2000/svg" width="560" height="288" class="svgbob">
  <style>.svgbob line, .svgbob path, .svgbob circle, .svgbob rect, .svgbob polygon {
  stroke: #838383;
  stroke-width: 2;
  stroke-opacity: 1;
  fill-opacity: 1;
  stroke-linecap: round;
  stroke-linejoin: miter;
}

.svgbob text {
  white-space: pre;
  fill: #838383;
  font-family: Iosevka Fixed, monospace;
  font-size: 14px;
}

.svgbob rect.backdrop {
  stroke: none;
  fill: rgba(0,0,0,0.0);
}

.svgbob .broken {
  stroke-dasharray: 8;
}

.svgbob .filled {
  fill: #838383;
}

.svgbob .bg_filled {
  fill: rgba(0,0,0,0.0);
  stroke-width: 1;
}

.svgbob .nofill {
  fill: rgba(0,0,0,0.0);
}

.svgbob .end_marked_arrow {
  marker-end: url(#arrow);
}

.svgbob .start_marked_arrow {
  marker-start: url(#arrow);
}

.svgbob .end_marked_diamond {
  marker-end: url(#diamond);
}

.svgbob .start_marked_diamond {
  marker-start: url(#diamond);
}

.svgbob .end_marked_circle {
  marker-end: url(#circle);
}

.svgbob .start_marked_circle {
  marker-start: url(#circle);
}

.svgbob .end_marked_open_circle {
  marker-end: url(#open_circle);
}

.svgbob .start_marked_open_circle {
  marker-start: url(#open_circle);
}

.svgbob .end_marked_big_open_circle {
  marker-end: url(#big_open_circle);
}

.svgbob .start_marked_big_open_circle {
  marker-start: url(#big_open_circle);
}

</style>
  <defs>
    <marker id="arrow" viewBox="-2 -2 8 8" refX="4" refY="2" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
      <polygon points="0,0 0,4 4,2 0,0"></polygon>
    </marker>
    <marker id="diamond" viewBox="-2 -2 8 8" refX="4" refY="2" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
      <polygon points="0,2 2,0 4,2 2,4 0,2"></polygon>
    </marker>
    <marker id="circle" viewBox="0 0 8 8" refX="4" refY="4" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
      <circle cx="4" cy="4" r="2" class="filled"></circle>
    </marker>
    <marker id="open_circle" viewBox="0 0 8 8" refX="4" refY="4" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
      <circle cx="4" cy="4" r="2" class="bg_filled"></circle>
    </marker>
    <marker id="big_open_circle" viewBox="0 0 8 8" refX="4" refY="4" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
      <circle cx="4" cy="4" r="3" class="bg_filled"></circle>
    </marker>
  </defs>
  <rect class="backdrop" x="0" y="0" width="560" height="288"></rect>
  <rect x="236" y="72" width="80" height="32" class="solid nofill" rx="4"></rect>
  <text x="250" y="92" >Network</text>
  <rect x="388" y="72" width="56" height="32" class="solid nofill" rx="4"></rect>
  <text x="402" y="92" >Loss</text>
  <rect x="20" y="136" width="56" height="32" class="solid nofill" rx="4"></rect>
  <text x="42" y="156" >x1</text>
  <rect x="20" y="200" width="56" height="32" class="solid nofill" rx="4"></rect>
  <text x="42" y="220" >x2</text>
  <text x="34" y="28" >Data</text>
  <text x="202" y="28" >GPU0</text>
  <polygon points="416,52 424,52 420,64" class="filled"></polygon>
  <text x="42" y="92" >x0</text>
  <polygon points="216,84 224,88 216,92" class="filled"></polygon>
  <polygon points="272,124 276,112 280,124" class="filled"></polygon>
  <g>
    <path d="M 8,8 A 4,4 0,0,0 4,12" class="nofill"></path>
    <line x1="4" y1="12" x2="4" y2="260" class="solid"></line>
    <line x1="8" y1="8" x2="88" y2="8" class="solid"></line>
    <path d="M 88,8 A 4,4 0,0,1 92,12" class="nofill"></path>
    <line x1="92" y1="12" x2="92" y2="80" class="solid"></line>
    <path d="M 4,260 A 4,4 0,0,0 8,264" class="nofill"></path>
    <line x1="8" y1="264" x2="88" y2="264" class="solid"></line>
    <line x1="92" y1="96" x2="92" y2="260" class="solid"></line>
    <path d="M 92,260 A 4,4 0,0,1 88,264" class="nofill"></path>
  </g>
  <g>
    <path d="M 192,8 A 4,4 0,0,0 188,12" class="nofill"></path>
    <line x1="188" y1="12" x2="188" y2="80" class="solid"></line>
    <line x1="192" y1="8" x2="540" y2="8" class="solid"></line>
    <path d="M 540,8 A 8,8 0,0,1 548,16" class="nofill"></path>
    <line x1="548" y1="16" x2="548" y2="160" class="solid"></line>
    <line x1="188" y1="96" x2="188" y2="164" class="solid"></line>
    <path d="M 188,164 A 4,4 0,0,0 192,168" class="nofill"></path>
    <line x1="192" y1="168" x2="540" y2="168" class="solid"></line>
    <path d="M 548,160 A 8,8 0,0,1 540,168" class="nofill"></path>
  </g>
  <g>
    <path d="M 280,24 A 4,4 0,0,0 276,28" class="nofill"></path>
    <line x1="276" y1="28" x2="276" y2="64" class="solid"></line>
    <line x1="280" y1="24" x2="416" y2="24" class="solid"></line>
    <path d="M 416,24 A 4,4 0,0,1 420,28" class="nofill"></path>
    <line x1="420" y1="28" x2="420" y2="52" class="solid"></line>
  </g>
  <g>
    <path d="M 24,72 A 4,4 0,0,0 20,76" class="nofill"></path>
    <line x1="20" y1="76" x2="20" y2="100" class="solid"></line>
    <line x1="24" y1="72" x2="72" y2="72" class="solid"></line>
    <path d="M 72,72 A 4,4 0,0,1 76,76" class="nofill"></path>
    <line x1="76" y1="76" x2="76" y2="100" class="solid"></line>
    <line x1="76" y1="88" x2="216" y2="88" class="solid"></line>
    <path d="M 20,100 A 4,4 0,0,0 24,104" class="nofill"></path>
    <line x1="24" y1="104" x2="72" y2="104" class="solid"></line>
    <path d="M 76,100 A 4,4 0,0,1 72,104" class="nofill"></path>
  </g>
  <g>
    <line x1="276" y1="124" x2="276" y2="148" class="solid"></line>
    <path d="M 276,148 A 4,4 0,0,0 280,152" class="nofill"></path>
    <line x1="280" y1="152" x2="416" y2="152" class="solid"></line>
    <line x1="420" y1="112" x2="420" y2="148" class="solid"></line>
    <path d="M 420,148 A 4,4 0,0,1 416,152" class="nofill"></path>
  </g>
</svg>

<svg xmlns="http://www.w3.org/2000/svg" width="496" height="576" class="svgbob">
  <style>.svgbob line, .svgbob path, .svgbob circle, .svgbob rect, .svgbob polygon {
  stroke: #838383;
  stroke-width: 2;
  stroke-opacity: 1;
  fill-opacity: 1;
  stroke-linecap: round;
  stroke-linejoin: miter;
}

.svgbob text {
  white-space: pre;
  fill: #838383;
  font-family: Iosevka Fixed, monospace;
  font-size: 14px;
}

.svgbob rect.backdrop {
  stroke: none;
  fill: rgba(0,0,0,0.0);
}

.svgbob .broken {
  stroke-dasharray: 8;
}

.svgbob .filled {
  fill: #838383;
}

.svgbob .bg_filled {
  fill: rgba(0,0,0,0.0);
  stroke-width: 1;
}

.svgbob .nofill {
  fill: rgba(0,0,0,0.0);
}

.svgbob .end_marked_arrow {
  marker-end: url(#arrow);
}

.svgbob .start_marked_arrow {
  marker-start: url(#arrow);
}

.svgbob .end_marked_diamond {
  marker-end: url(#diamond);
}

.svgbob .start_marked_diamond {
  marker-start: url(#diamond);
}

.svgbob .end_marked_circle {
  marker-end: url(#circle);
}

.svgbob .start_marked_circle {
  marker-start: url(#circle);
}

.svgbob .end_marked_open_circle {
  marker-end: url(#open_circle);
}

.svgbob .start_marked_open_circle {
  marker-start: url(#open_circle);
}

.svgbob .end_marked_big_open_circle {
  marker-end: url(#big_open_circle);
}

.svgbob .start_marked_big_open_circle {
  marker-start: url(#big_open_circle);
}

</style>
  <defs>
    <marker id="arrow" viewBox="-2 -2 8 8" refX="4" refY="2" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
      <polygon points="0,0 0,4 4,2 0,0"></polygon>
    </marker>
    <marker id="diamond" viewBox="-2 -2 8 8" refX="4" refY="2" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
      <polygon points="0,2 2,0 4,2 2,4 0,2"></polygon>
    </marker>
    <marker id="circle" viewBox="0 0 8 8" refX="4" refY="4" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
      <circle cx="4" cy="4" r="2" class="filled"></circle>
    </marker>
    <marker id="open_circle" viewBox="0 0 8 8" refX="4" refY="4" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
      <circle cx="4" cy="4" r="2" class="bg_filled"></circle>
    </marker>
    <marker id="big_open_circle" viewBox="0 0 8 8" refX="4" refY="4" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
      <circle cx="4" cy="4" r="3" class="bg_filled"></circle>
    </marker>
  </defs>
  <rect class="backdrop" x="0" y="0" width="496" height="576"></rect>
  <rect x="388" y="72" width="56" height="32" class="solid nofill" rx="4"></rect>
  <text x="402" y="92" >Loss</text>
  <rect x="236" y="264" width="80" height="32" class="solid nofill" rx="4"></rect>
  <text x="250" y="284" >Network</text>
  <rect x="388" y="264" width="56" height="32" class="solid nofill" rx="4"></rect>
  <text x="402" y="284" >Loss</text>
  <rect x="236" y="456" width="80" height="32" class="solid nofill" rx="4"></rect>
  <text x="250" y="476" >Network</text>
  <rect x="388" y="456" width="56" height="32" class="solid nofill" rx="4"></rect>
  <text x="402" y="476" >Loss</text>
  <text x="34" y="28" >Data</text>
  <text x="202" y="28" >GPU0</text>
  <text x="42" y="92" >x0</text>
  <polygon points="216,84 224,88 216,92" class="filled"></polygon>
  <text x="250" y="92" >Network</text>
  <polygon points="376,84 384,88 376,92" class="filled"></polygon>
  <line x1="92" y1="96" x2="92" y2="272" class="solid"></line>
  <polygon points="272,124 276,112 280,124" class="filled"></polygon>
  <polygon points="321,124 314,116 314,127" class="filled"></polygon>
  <text x="202" y="220" >GPU1</text>
  <polygon points="416,244 424,244 420,256" class="filled"></polygon>
  <text x="42" y="284" >x1</text>
  <polygon points="216,276 224,280 216,284" class="filled"></polygon>
  <line x1="92" y1="288" x2="92" y2="464" class="solid"></line>
  <polygon points="272,316 276,304 280,316" class="filled"></polygon>
  <text x="202" y="412" >GPU2</text>
  <polygon points="416,436 424,436 420,448" class="filled"></polygon>
  <text x="42" y="476" >x2</text>
  <polygon points="216,468 224,472 216,476" class="filled"></polygon>
  <polygon points="272,508 276,496 280,508" class="filled"></polygon>
  <g>
    <path d="M 8,8 A 4,4 0,0,0 4,12" class="nofill"></path>
    <line x1="4" y1="12" x2="4" y2="548" class="solid"></line>
    <line x1="8" y1="8" x2="88" y2="8" class="solid"></line>
    <path d="M 88,8 A 4,4 0,0,1 92,12" class="nofill"></path>
    <line x1="92" y1="12" x2="92" y2="80" class="solid"></line>
    <path d="M 4,548 A 4,4 0,0,0 8,552" class="nofill"></path>
    <line x1="8" y1="552" x2="88" y2="552" class="solid"></line>
    <line x1="92" y1="480" x2="92" y2="548" class="solid"></line>
    <path d="M 92,548 A 4,4 0,0,1 88,552" class="nofill"></path>
  </g>
  <g>
    <path d="M 192,8 A 4,4 0,0,0 188,12" class="nofill"></path>
    <line x1="188" y1="12" x2="188" y2="80" class="solid"></line>
    <line x1="192" y1="8" x2="480" y2="8" class="solid"></line>
    <path d="M 480,8 A 4,4 0,0,1 484,12" class="nofill"></path>
    <line x1="484" y1="12" x2="484" y2="164" class="solid"></line>
    <line x1="188" y1="96" x2="188" y2="164" class="solid"></line>
    <path d="M 188,164 A 4,4 0,0,0 192,168" class="nofill"></path>
    <line x1="192" y1="168" x2="480" y2="168" class="solid"></line>
    <path d="M 484,164 A 4,4 0,0,1 480,168" class="nofill"></path>
  </g>
  <g>
    <path d="M 24,72 A 4,4 0,0,0 20,76" class="nofill"></path>
    <line x1="20" y1="76" x2="20" y2="100" class="solid"></line>
    <line x1="24" y1="72" x2="72" y2="72" class="solid"></line>
    <path d="M 72,72 A 4,4 0,0,1 76,76" class="nofill"></path>
    <line x1="76" y1="76" x2="76" y2="100" class="solid"></line>
    <line x1="76" y1="88" x2="216" y2="88" class="solid"></line>
    <path d="M 20,100 A 4,4 0,0,0 24,104" class="nofill"></path>
    <line x1="24" y1="104" x2="72" y2="104" class="solid"></line>
    <path d="M 76,100 A 4,4 0,0,1 72,104" class="nofill"></path>
  </g>
  <g>
    <path d="M 240,72 A 4,4 0,0,0 236,76" class="nofill"></path>
    <line x1="236" y1="76" x2="236" y2="100" class="solid"></line>
    <line x1="240" y1="72" x2="312" y2="72" class="solid"></line>
    <path d="M 312,72 A 4,4 0,0,1 316,76" class="nofill"></path>
    <line x1="316" y1="76" x2="316" y2="100" class="solid"></line>
    <line x1="316" y1="88" x2="376" y2="88" class="solid"></line>
    <path d="M 236,100 A 4,4 0,0,0 240,104" class="nofill"></path>
    <line x1="240" y1="104" x2="312" y2="104" class="solid"></line>
    <path d="M 316,100 A 4,4 0,0,1 312,104" class="nofill"></path>
  </g>
  <g>
    <line x1="276" y1="124" x2="276" y2="148" class="solid"></line>
    <path d="M 276,148 A 4,4 0,0,0 280,152" class="nofill"></path>
    <line x1="280" y1="152" x2="416" y2="152" class="solid"></line>
    <line x1="420" y1="112" x2="420" y2="148" class="solid"></line>
    <path d="M 420,148 A 4,4 0,0,1 416,152" class="nofill"></path>
  </g>
  <g>
    <line x1="318" y1="124" x2="328" y2="144" class="solid"></line>
    <line x1="384" y1="112" x2="368" y2="144" class="solid"></line>
    <line x1="328" y1="144" x2="368" y2="144" class="solid"></line>
  </g>
  <g>
    <path d="M 192,200 A 4,4 0,0,0 188,204" class="nofill"></path>
    <line x1="188" y1="204" x2="188" y2="272" class="solid"></line>
    <line x1="192" y1="200" x2="480" y2="200" class="solid"></line>
    <path d="M 480,200 A 4,4 0,0,1 484,204" class="nofill"></path>
    <line x1="484" y1="204" x2="484" y2="356" class="solid"></line>
    <line x1="188" y1="288" x2="188" y2="356" class="solid"></line>
    <path d="M 188,356 A 4,4 0,0,0 192,360" class="nofill"></path>
    <line x1="192" y1="360" x2="480" y2="360" class="solid"></line>
    <path d="M 484,356 A 4,4 0,0,1 480,360" class="nofill"></path>
  </g>
  <g>
    <path d="M 280,216 A 4,4 0,0,0 276,220" class="nofill"></path>
    <line x1="276" y1="220" x2="276" y2="256" class="solid"></line>
    <line x1="280" y1="216" x2="416" y2="216" class="solid"></line>
    <path d="M 416,216 A 4,4 0,0,1 420,220" class="nofill"></path>
    <line x1="420" y1="220" x2="420" y2="244" class="solid"></line>
  </g>
  <g>
    <path d="M 24,264 A 4,4 0,0,0 20,268" class="nofill"></path>
    <line x1="20" y1="268" x2="20" y2="292" class="solid"></line>
    <line x1="24" y1="264" x2="72" y2="264" class="solid"></line>
    <path d="M 72,264 A 4,4 0,0,1 76,268" class="nofill"></path>
    <line x1="76" y1="268" x2="76" y2="292" class="solid"></line>
    <line x1="76" y1="280" x2="216" y2="280" class="solid"></line>
    <path d="M 20,292 A 4,4 0,0,0 24,296" class="nofill"></path>
    <line x1="24" y1="296" x2="72" y2="296" class="solid"></line>
    <path d="M 76,292 A 4,4 0,0,1 72,296" class="nofill"></path>
  </g>
  <g>
    <line x1="276" y1="316" x2="276" y2="340" class="solid"></line>
    <path d="M 276,340 A 4,4 0,0,0 280,344" class="nofill"></path>
    <line x1="280" y1="344" x2="416" y2="344" class="solid"></line>
    <line x1="420" y1="304" x2="420" y2="340" class="solid"></line>
    <path d="M 420,340 A 4,4 0,0,1 416,344" class="nofill"></path>
  </g>
  <g>
    <path d="M 192,392 A 4,4 0,0,0 188,396" class="nofill"></path>
    <line x1="188" y1="396" x2="188" y2="464" class="solid"></line>
    <line x1="192" y1="392" x2="480" y2="392" class="solid"></line>
    <path d="M 480,392 A 4,4 0,0,1 484,396" class="nofill"></path>
    <line x1="484" y1="396" x2="484" y2="548" class="solid"></line>
    <line x1="188" y1="480" x2="188" y2="548" class="solid"></line>
    <path d="M 188,548 A 4,4 0,0,0 192,552" class="nofill"></path>
    <line x1="192" y1="552" x2="480" y2="552" class="solid"></line>
    <path d="M 484,548 A 4,4 0,0,1 480,552" class="nofill"></path>
  </g>
  <g>
    <path d="M 280,408 A 4,4 0,0,0 276,412" class="nofill"></path>
    <line x1="276" y1="412" x2="276" y2="448" class="solid"></line>
    <line x1="280" y1="408" x2="416" y2="408" class="solid"></line>
    <path d="M 416,408 A 4,4 0,0,1 420,412" class="nofill"></path>
    <line x1="420" y1="412" x2="420" y2="436" class="solid"></line>
  </g>
  <g>
    <path d="M 24,456 A 4,4 0,0,0 20,460" class="nofill"></path>
    <line x1="20" y1="460" x2="20" y2="484" class="solid"></line>
    <line x1="24" y1="456" x2="72" y2="456" class="solid"></line>
    <path d="M 72,456 A 4,4 0,0,1 76,460" class="nofill"></path>
    <line x1="76" y1="460" x2="76" y2="484" class="solid"></line>
    <line x1="76" y1="472" x2="216" y2="472" class="solid"></line>
    <path d="M 20,484 A 4,4 0,0,0 24,488" class="nofill"></path>
    <line x1="24" y1="488" x2="72" y2="488" class="solid"></line>
    <path d="M 76,484 A 4,4 0,0,1 72,488" class="nofill"></path>
  </g>
  <g>
    <line x1="276" y1="508" x2="276" y2="532" class="solid"></line>
    <path d="M 276,532 A 4,4 0,0,0 280,536" class="nofill"></path>
    <line x1="280" y1="536" x2="416" y2="536" class="solid"></line>
    <line x1="420" y1="496" x2="420" y2="532" class="solid"></line>
    <path d="M 420,532 A 4,4 0,0,1 416,536" class="nofill"></path>
  </g>
</svg>

<svg xmlns="http://www.w3.org/2000/svg" width="576" height="704" class="svgbob">
  <style>.svgbob line, .svgbob path, .svgbob circle, .svgbob rect, .svgbob polygon {
  stroke: #838383;
  stroke-width: 2;
  stroke-opacity: 1;
  fill-opacity: 1;
  stroke-linecap: round;
  stroke-linejoin: miter;
}

.svgbob text {
  white-space: pre;
  fill: #838383;
  font-family: Iosevka Fixed, monospace;
  font-size: 14px;
}

.svgbob rect.backdrop {
  stroke: none;
  fill: rgba(0,0,0,0.0);
}

.svgbob .broken {
  stroke-dasharray: 8;
}

.svgbob .filled {
  fill: #838383;
}

.svgbob .bg_filled {
  fill: rgba(0,0,0,0.0);
  stroke-width: 1;
}

.svgbob .nofill {
  fill: rgba(0,0,0,0.0);
}

.svgbob .end_marked_arrow {
  marker-end: url(#arrow);
}

.svgbob .start_marked_arrow {
  marker-start: url(#arrow);
}

.svgbob .end_marked_diamond {
  marker-end: url(#diamond);
}

.svgbob .start_marked_diamond {
  marker-start: url(#diamond);
}

.svgbob .end_marked_circle {
  marker-end: url(#circle);
}

.svgbob .start_marked_circle {
  marker-start: url(#circle);
}

.svgbob .end_marked_open_circle {
  marker-end: url(#open_circle);
}

.svgbob .start_marked_open_circle {
  marker-start: url(#open_circle);
}

.svgbob .end_marked_big_open_circle {
  marker-end: url(#big_open_circle);
}

.svgbob .start_marked_big_open_circle {
  marker-start: url(#big_open_circle);
}

.svgbob .r{ 
    fill: #FF8180;
 }
.svgbob .g{ 
    fill: #97E6A5;
 }
.svgbob .b{ 
    fill: #7DCAFF;
 }
.svgbob .t{ 
    fill: none;
    stroke: #838383;
    color: #838383;
 }
.svgbob .r1{ 
    fill: papayawhip;
 }
.svgbob .r2{ 
    fill: crimson;
 }
.svgbob .a{ 
    stroke-dasharray: 8;
    fill: lightblue;
 }
.svgbob .bigrect{ 
    fill: yellow;
    stroke: red;
 }
.svgbob .red{ 
    fill:red;
    stroke:blue;
 }
.svgbob .container{ 
    fill: rgba(100,100,100, 0.1);
    stroke: #000;
 }</style>
  <defs>
    <marker id="arrow" viewBox="-2 -2 8 8" refX="4" refY="2" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
      <polygon points="0,0 0,4 4,2 0,0"></polygon>
    </marker>
    <marker id="diamond" viewBox="-2 -2 8 8" refX="4" refY="2" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
      <polygon points="0,2 2,0 4,2 2,4 0,2"></polygon>
    </marker>
    <marker id="circle" viewBox="0 0 8 8" refX="4" refY="4" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
      <circle cx="4" cy="4" r="2" class="filled"></circle>
    </marker>
    <marker id="open_circle" viewBox="0 0 8 8" refX="4" refY="4" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
      <circle cx="4" cy="4" r="2" class="bg_filled"></circle>
    </marker>
    <marker id="big_open_circle" viewBox="0 0 8 8" refX="4" refY="4" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
      <circle cx="4" cy="4" r="3" class="bg_filled"></circle>
    </marker>
  </defs>
  <rect class="backdrop" x="0" y="0" width="576" height="704"></rect>
  <rect x="4" y="40" width="560" height="448" class="broken nofill g container" rx="4"></rect>
  <rect x="36" y="72" width="56" height="32" class="solid nofill r1" rx="0"></rect>
  <rect x="124" y="72" width="56" height="32" class="solid nofill r2" rx="4"></rect>
  <rect x="308" y="72" width="216" height="160" class="solid nofill bigrect" rx="4"></rect>
  <circle cx="68" cy="304" r="36" class="nofill a"></circle>
  <text x="66" y="300" >8</text>
  <circle cx="160" cy="304" r="40" class="nofill b"></circle>
  <text x="154" y="300" >9</text>
  <circle cx="264" cy="312" r="44" class="nofill red"></circle>
  <text x="258" y="316" >10</text>
  <circle cx="384" cy="312" r="44" class="nofill a b"></circle>
  <text x="378" y="316" >11</text>
  <circle cx="96" cy="416" r="52" class="nofill c"></circle>
  <text x="90" y="412" >12</text>
  <line x1="16" y1="8" x2="4" y2="8" class="solid end_marked_open_circle"></line>
  <polygon points="16,4 24,8 16,12" class="filled"></polygon>
  <text x="34" y="12" >Latest</text>
  <text x="90" y="12" >addition:</text>
  <text x="170" y="12" >Styling</text>
  <text x="234" y="12" >of</text>
  <text x="258" y="12" >tagged</text>
  <text x="314" y="12" >shapes</text>
  <text x="26" y="572" >Advantages:</text>
  <line x1="8" y1="600" x2="20" y2="600" class="solid end_marked_circle"></line>
  <line x1="16" y1="600" x2="20" y2="600" class="solid"></line>
  <text x="34" y="604" >Plain</text>
  <text x="82" y="604" >text</text>
  <text x="122" y="604" >format</text>
  <text x="34" y="620" >Ultimately</text>
  <text x="122" y="620" >portable,</text>
  <text x="34" y="636" >Degrades</text>
  <text x="106" y="636" >gracefully</text>
  <text x="34" y="652" >Even</text>
  <text x="74" y="652" >when</text>
  <text x="114" y="652" >not</text>
  <text x="146" y="652" >using</text>
  <text x="194" y="652" >a</text>
  <text x="210" y="652" >graphical</text>
  <text x="290" y="652" >renderer,</text>
  <text x="370" y="652" >it</text>
  <text x="394" y="652" >would</text>
  <text x="442" y="652" >still</text>
  <text x="490" y="652" >looks</text>
  <text x="34" y="668" >as</text>
  <text x="58" y="668" >text</text>
  <text x="98" y="668" >based</text>
  <text x="146" y="668" >diagrams.</text>
  <text x="226" y="668" >Paste</text>
  <text x="274" y="668" >the</text>
  <text x="306" y="668" >text</text>
  <text x="346" y="668" >in</text>
  <text x="370" y="668" >your</text>
  <text x="410" y="668" >source</text>
  <text x="466" y="668" >code.</text>
  <text x="34" y="684" >Easiest</text>
  <text x="98" y="684" >to</text>
  <text x="122" y="684" >use.</text>
  <text x="162" y="684" >Anyone</text>
  <text x="218" y="684" >knows</text>
  <text x="266" y="684" >how</text>
  <text x="298" y="684" >to</text>
  <text x="322" y="684" >edit</text>
  <text x="362" y="684" >text.</text>
  <text x="202" y="620" >backward</text>
  <text x="274" y="620" >compatible</text>
  <text x="362" y="620" >and</text>
  <text x="394" y="620" >future</text>
  <text x="450" y="620" >proof.</text>
  <line x1="8" y1="632" x2="20" y2="632" class="solid end_marked_circle"></line>
  <line x1="16" y1="632" x2="20" y2="632" class="solid"></line>
  <text x="538" y="652" >good</text>
  <line x1="8" y1="680" x2="20" y2="680" class="solid end_marked_circle"></line>
  <line x1="16" y1="680" x2="20" y2="680" class="solid"></line>
  <g>
    <line x1="48" y1="128" x2="104" y2="128" class="solid"></line>
    <line x1="48" y1="128" x2="32" y2="160" class="solid"></line>
    <line x1="104" y1="128" x2="120" y2="160" class="solid"></line>
    <line x1="32" y1="160" x2="48" y2="192" class="solid"></line>
    <line x1="120" y1="160" x2="104" y2="192" class="solid"></line>
    <line x1="48" y1="192" x2="104" y2="192" class="solid"></line>
  </g>
  <g>
    <path d="M 224,72 A 8,8 0,0,0 218,76" class="nofill"></path>
    <line x1="218" y1="76" x2="216" y2="80" class="solid"></line>
    <line x1="224" y1="72" x2="272" y2="72" class="solid"></line>
    <path d="M 272,72 A 8,8 0,0,1 278,76" class="nofill"></path>
    <line x1="278" y1="76" x2="280" y2="80" class="solid"></line>
    <path d="M 216,80 A 16,16 0,0,0 216,96" class="nofill"></path>
    <path d="M 280,80 A 16,16 0,0,1 280,96" class="nofill"></path>
    <line x1="216" y1="96" x2="218" y2="100" class="solid"></line>
    <path d="M 218,100 A 8,8 0,0,0 224,104" class="nofill"></path>
    <line x1="224" y1="104" x2="272" y2="104" class="solid"></line>
    <line x1="280" y1="96" x2="278" y2="100" class="solid"></line>
    <path d="M 278,100 A 8,8 0,0,1 272,104" class="nofill"></path>
  </g>
  <g>
    <line x1="200" y1="128" x2="264" y2="128" class="solid"></line>
    <line x1="200" y1="128" x2="208" y2="144" class="solid"></line>
    <line x1="264" y1="128" x2="272" y2="144" class="solid"></line>
    <path d="M 208,144 A 16,16 0,0,1 208,160" class="nofill"></path>
    <path d="M 272,144 A 16,16 0,0,1 272,160" class="nofill"></path>
    <line x1="208" y1="160" x2="200" y2="176" class="solid"></line>
    <line x1="200" y1="176" x2="264" y2="176" class="solid"></line>
    <line x1="272" y1="160" x2="264" y2="176" class="solid"></line>
  </g>
  <g>
    <line x1="168" y1="128" x2="152" y2="160" class="solid"></line>
    <line x1="168" y1="128" x2="184" y2="160" class="solid"></line>
    <line x1="152" y1="160" x2="168" y2="192" class="solid"></line>
    <line x1="184" y1="160" x2="168" y2="192" class="solid"></line>
  </g>
  <g>
    <path d="M 184,392 A 3,3 0,0,0 182,396" class="nofill"></path>
    <line x1="182" y1="396" x2="204" y2="440" class="solid"></line>
    <line x1="184" y1="392" x2="224" y2="392" class="solid"></line>
    <path d="M 224,392 A 3,3 0,0,1 226,396" class="nofill"></path>
    <line x1="226" y1="396" x2="204" y2="440" class="solid"></line>
  </g>
  <g>
    <path d="M 248,392 A 3,3 0,0,0 246,396" class="nofill"></path>
    <line x1="246" y1="396" x2="266" y2="436" class="solid"></line>
    <line x1="248" y1="392" x2="280" y2="392" class="solid"></line>
    <path d="M 280,392 A 8,8 0,0,1 286,396" class="nofill"></path>
    <line x1="286" y1="396" x2="306" y2="436" class="solid"></line>
    <path d="M 266,436 A 8,8 0,0,0 272,440" class="nofill"></path>
    <line x1="272" y1="440" x2="304" y2="440" class="solid"></line>
    <path d="M 306,436 A 3,3 0,0,1 304,440" class="nofill"></path>
  </g>
  <g>
    <path d="M 24,552 A 8,8 0,0,0 18,556" class="nofill"></path>
    <line x1="18" y1="556" x2="6" y2="580" class="solid"></line>
    <line x1="24" y1="552" x2="128" y2="552" class="solid"></line>
    <path d="M 128,552 A 3,3 0,0,1 130,556" class="nofill"></path>
    <line x1="130" y1="556" x2="118" y2="580" class="solid"></line>
    <path d="M 6,580 A 3,3 0,0,0 8,584" class="nofill"></path>
    <line x1="8" y1="584" x2="112" y2="584" class="solid"></line>
    <path d="M 118,580 A 8,8 0,0,1 112,584" class="nofill"></path>
  </g>
</svg>

<svg xmlns="http://www.w3.org/2000/svg" width="760" height="7520" class="svgbob">
  <style>.svgbob line, .svgbob path, .svgbob circle, .svgbob rect, .svgbob polygon {
  stroke: #838383;
  stroke-width: 2;
  stroke-opacity: 1;
  fill-opacity: 1;
  stroke-linecap: round;
  stroke-linejoin: miter;
}

.svgbob text {
  white-space: pre;
  fill: #838383;
  font-family: Iosevka Fixed, monospace;
  font-size: 14px;
}

.svgbob rect.backdrop {
  stroke: none;
  fill: rgba(0,0,0,0.0);
}

.svgbob .broken {
  stroke-dasharray: 8;
}

.svgbob .filled {
  fill: #838383;
}

.svgbob .bg_filled {
  fill: rgba(0,0,0,0.0);
  stroke-width: 1;
}

.svgbob .nofill {
  fill: rgba(0,0,0,0.0);
}

.svgbob .end_marked_arrow {
  marker-end: url(#arrow);
}

.svgbob .start_marked_arrow {
  marker-start: url(#arrow);
}

.svgbob .end_marked_diamond {
  marker-end: url(#diamond);
}

.svgbob .start_marked_diamond {
  marker-start: url(#diamond);
}

.svgbob .end_marked_circle {
  marker-end: url(#circle);
}

.svgbob .start_marked_circle {
  marker-start: url(#circle);
}

.svgbob .end_marked_open_circle {
  marker-end: url(#open_circle);
}

.svgbob .start_marked_open_circle {
  marker-start: url(#open_circle);
}

.svgbob .end_marked_big_open_circle {
  marker-end: url(#big_open_circle);
}

.svgbob .start_marked_big_open_circle {
  marker-start: url(#big_open_circle);
}

</style>
  <defs>
    <marker id="arrow" viewBox="-2 -2 8 8" refX="4" refY="2" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
      <polygon points="0,0 0,4 4,2 0,0"></polygon>
    </marker>
    <marker id="diamond" viewBox="-2 -2 8 8" refX="4" refY="2" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
      <polygon points="0,2 2,0 4,2 2,4 0,2"></polygon>
    </marker>
    <marker id="circle" viewBox="0 0 8 8" refX="4" refY="4" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
      <circle cx="4" cy="4" r="2" class="filled"></circle>
    </marker>
    <marker id="open_circle" viewBox="0 0 8 8" refX="4" refY="4" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
      <circle cx="4" cy="4" r="2" class="bg_filled"></circle>
    </marker>
    <marker id="big_open_circle" viewBox="0 0 8 8" refX="4" refY="4" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
      <circle cx="4" cy="4" r="3" class="bg_filled"></circle>
    </marker>
  </defs>
  <rect class="backdrop" x="0" y="0" width="760" height="7520"></rect>
  <rect x="68" y="312" width="56" height="32" class="solid nofill" rx="0"></rect>
  <rect x="156" y="312" width="56" height="32" class="solid nofill" rx="4"></rect>
  <path d="M 88,468 A 20,20 0,0,0 88,508" class="nofill"></path>
  <circle cx="72" cy="536" r="4" class="nofill"></circle>
  <circle cx="444" cy="568" r="24" class="nofill"></circle>
  <circle cx="520" cy="568" r="28" class="nofill"></circle>
  <circle cx="156" cy="568" r="16" class="nofill"></circle>
  <circle cx="208" cy="568" r="20" class="nofill"></circle>
  <circle cx="264" cy="568" r="20" class="nofill"></circle>
  <path d="M 272,560 A 16,16 0,0,0 272,576" class="nofill"></path>
  <circle cx="360" cy="568" r="20" class="nofill"></circle>
  <circle cx="76" cy="568" r="8" class="nofill"></circle>
  <circle cx="100" cy="656" r="36" class="nofill"></circle>
  <circle cx="192" cy="656" r="40" class="nofill"></circle>
  <circle cx="296" cy="664" r="44" class="nofill"></circle>
  <circle cx="420" cy="664" r="48" class="nofill"></circle>
  <circle cx="128" cy="768" r="52" class="nofill"></circle>
  <rect x="146" y="1176" width="28" height="36" class="solid nofill" rx="0"></rect>
  <rect x="150" y="1180" width="12" height="12" class="solid nofill" rx="0"></rect>
  <rect x="166" y="1180" width="4" height="12" class="solid nofill" rx="0"></rect>
  <rect x="150" y="1196" width="12" height="12" class="solid nofill" rx="0"></rect>
  <rect x="166" y="1196" width="4" height="12" class="solid nofill" rx="0"></rect>
  <circle cx="72" cy="1376" r="40" class="nofill"></circle>
  <path d="M 200,1336 A 40,40 0,0,0 160,1376" class="nofill"></path>
  <path d="M 216,1336 A 40,40 0,0,1 256,1376" class="nofill"></path>
  <path d="M 312,1376 A 40,40 0,0,1 392,1376" class="nofill"></path>
  <path d="M 480,1352 A 40,40 0,0,0 480,1432" class="nofill"></path>
  <path d="M 504,1352 A 40,40 0,0,1 504,1432" class="nofill"></path>
  <path d="M 160,1392 A 40,40 0,0,0 200,1432" class="nofill"></path>
  <path d="M 256,1392 A 40,40 0,0,1 216,1432" class="nofill"></path>
  <path d="M 312,1392 A 40,40 0,0,0 392,1392" class="nofill"></path>
  <path d="M 112,1488 A 40,40 0,1,0 72,1528" class="nofill"></path>
  <path d="M 208,1448 A 40,40 0,1,0 248,1488" class="nofill"></path>
  <path d="M 352,1448 A 40,40 0,1,1 312,1488" class="nofill"></path>
  <path d="M 456,1488 A 40,40 0,1,1 496,1528" class="nofill"></path>
  <circle cx="516" cy="2168" r="16" class="nofill"></circle>
  <line x1="520" y1="2160" x2="512" y2="2176" class="solid"></line>
  <rect x="92" y="2296" width="104" height="32" class="solid nofill" rx="4"></rect>
  <rect x="228" y="2296" width="96" height="32" class="solid nofill" rx="4"></rect>
  <rect x="364" y="2296" width="48" height="32" class="solid nofill" rx="4"></rect>
  <text x="378" y="2316" >MMU</text>
  <rect x="212" y="2520" width="16" height="16" class="solid nofill" rx="4"></rect>
  <rect x="212" y="2712" width="16" height="16" class="solid nofill" rx="4"></rect>
  <rect x="108" y="4456" width="16" height="16" class="solid nofill" rx="4"></rect>
  <rect x="140" y="4456" width="16" height="16" class="solid nofill" rx="4"></rect>
  <rect x="172" y="4456" width="16" height="16" class="solid nofill" rx="4"></rect>
  <rect x="308" y="4408" width="32" height="16" class="solid nofill" rx="4"></rect>
  <rect x="364" y="4408" width="32" height="16" class="solid nofill" rx="4"></rect>
  <rect x="412" y="4408" width="32" height="16" class="solid nofill" rx="4"></rect>
  <rect x="468" y="4408" width="32" height="16" class="solid nofill" rx="4"></rect>
  <rect x="540" y="4408" width="32" height="16" class="solid nofill" rx="4"></rect>
  <rect x="308" y="4472" width="104" height="32" class="solid nofill" rx="4"></rect>
  <text x="322" y="4492" >Filesystem</text>
  <rect x="444" y="4472" width="96" height="32" class="solid nofill" rx="4"></rect>
  <text x="458" y="4492" >Scheduler</text>
  <rect x="340" y="4552" width="40" height="32" class="solid nofill" rx="4"></rect>
  <text x="354" y="4572" >IO</text>
  <rect x="524" y="4552" width="80" height="32" class="solid nofill" rx="4"></rect>
  <text x="538" y="4572" >Network</text>
  <rect x="308" y="4632" width="320" height="32" class="solid nofill" rx="4"></rect>
  <text x="458" y="4652" >HAL</text>
  <rect x="596" y="4408" width="32" height="16" class="solid nofill" rx="4"></rect>
  <rect x="580" y="4472" width="48" height="32" class="solid nofill" rx="4"></rect>
  <text x="594" y="4492" >MMU</text>
  <rect x="580" y="4800" width="24" height="48" class="solid nofill" rx="0"></rect>
  <line x1="584" y1="4806" x2="600" y2="4806" class="solid"></line>
  <line x1="584" y1="4810" x2="600" y2="4810" class="solid"></line>
  <rect x="36" y="6856" width="560" height="448" class="broken nofill container" rx="4"></rect>
  <rect x="68" y="6888" width="56" height="32" class="solid nofill r1" rx="0"></rect>
  <rect x="156" y="6888" width="56" height="32" class="solid nofill r2" rx="4"></rect>
  <rect x="340" y="6888" width="216" height="160" class="solid nofill bigrect" rx="4"></rect>
  <circle cx="100" cy="7120" r="36" class="nofill a"></circle>
  <text x="98" y="7116" >8</text>
  <circle cx="192" cy="7120" r="40" class="nofill b"></circle>
  <text x="186" y="7116" >9</text>
  <circle cx="296" cy="7128" r="44" class="nofill red"></circle>
  <text x="290" y="7132" >10</text>
  <circle cx="416" cy="7128" r="44" class="nofill a b"></circle>
  <text x="410" y="7132" >11</text>
  <circle cx="128" cy="7232" r="52" class="nofill c"></circle>
  <text x="122" y="7228" >12</text>
  <text x="2" y="12" >Svgbob</text>
  <text x="58" y="12" >is</text>
  <text x="82" y="12" >a</text>
  <text x="98" y="12" >diagramming</text>
  <text x="194" y="12" >model</text>
  <text x="2" y="28" >which</text>
  <text x="50" y="28" >uses</text>
  <text x="90" y="28" >a</text>
  <text x="106" y="28" >set</text>
  <text x="138" y="28" >of</text>
  <text x="162" y="28" >typing</text>
  <text x="218" y="28" >characters</text>
  <text x="2" y="44" >to</text>
  <text x="26" y="44" >approximate</text>
  <text x="122" y="44" >the</text>
  <text x="154" y="44" >intended</text>
  <text x="226" y="44" >shape.</text>
  <line x1="58" y1="76" x2="36" y2="120" class="solid end_marked_circle"></line>
  <line x1="52" y1="88" x2="68" y2="88" class="solid end_marked_open_circle"></line>
  <line x1="72" y1="88" x2="104" y2="88" class="solid"></line>
  <polygon points="88,100 96,104 88,108" class="filled"></polygon>
  <text x="2" y="188" >It</text>
  <text x="26" y="188" >uses</text>
  <text x="66" y="188" >a</text>
  <text x="2" y="204" >which</text>
  <text x="50" y="204" >are</text>
  <text x="82" y="188" >combination</text>
  <text x="178" y="188" >of</text>
  <text x="202" y="188" >characters</text>
  <text x="82" y="204" >readily</text>
  <text x="146" y="204" >available</text>
  <text x="226" y="204" >on</text>
  <text x="250" y="204" >your</text>
  <text x="290" y="204" >keyboards.</text>
  <text x="2" y="236" >What</text>
  <text x="42" y="236" >can</text>
  <text x="74" y="236" >it</text>
  <text x="98" y="236" >do?</text>
  <line x1="24" y1="264" x2="12" y2="264" class="solid end_marked_open_circle"></line>
  <polygon points="24,260 32,264 24,268" class="filled"></polygon>
  <text x="50" y="268" >Basic</text>
  <text x="98" y="268" >shapes</text>
  <line x1="56" y1="280" x2="64" y2="280" class="solid"></line>
  <line x1="72" y1="280" x2="80" y2="280" class="solid"></line>
  <line x1="88" y1="280" x2="96" y2="280" class="solid"></line>
  <line x1="104" y1="280" x2="112" y2="280" class="solid"></line>
  <line x1="120" y1="280" x2="128" y2="280" class="solid"></line>
  <line x1="136" y1="280" x2="144" y2="280" class="solid"></line>
  <line x1="152" y1="280" x2="160" y2="280" class="solid"></line>
  <line x1="168" y1="280" x2="176" y2="280" class="solid"></line>
  <line x1="184" y1="280" x2="192" y2="280" class="solid"></line>
  <line x1="200" y1="280" x2="208" y2="280" class="solid"></line>
  <line x1="216" y1="280" x2="224" y2="280" class="solid"></line>
  <line x1="232" y1="280" x2="240" y2="280" class="solid"></line>
  <line x1="248" y1="280" x2="256" y2="280" class="solid"></line>
  <line x1="264" y1="280" x2="272" y2="280" class="solid"></line>
  <line x1="280" y1="280" x2="288" y2="280" class="solid"></line>
  <line x1="296" y1="280" x2="304" y2="280" class="solid"></line>
  <line x1="312" y1="280" x2="320" y2="280" class="solid"></line>
  <line x1="328" y1="280" x2="336" y2="280" class="solid"></line>
  <line x1="344" y1="280" x2="352" y2="280" class="solid"></line>
  <line x1="360" y1="280" x2="368" y2="280" class="solid"></line>
  <line x1="376" y1="280" x2="384" y2="280" class="solid"></line>
  <line x1="392" y1="280" x2="400" y2="280" class="solid"></line>
  <line x1="408" y1="280" x2="416" y2="280" class="solid"></line>
  <line x1="424" y1="280" x2="432" y2="280" class="solid"></line>
  <line x1="440" y1="280" x2="448" y2="280" class="solid"></line>
  <line x1="456" y1="280" x2="464" y2="280" class="solid"></line>
  <text x="458" y="300" >.</text>
  <text x="426" y="332" >.</text>
  <text x="490" y="332" >.</text>
  <text x="458" y="364" >.</text>
  <line x1="472" y1="280" x2="480" y2="280" class="solid"></line>
  <line x1="488" y1="280" x2="496" y2="280" class="solid"></line>
  <line x1="504" y1="280" x2="512" y2="280" class="solid"></line>
  <line x1="520" y1="280" x2="528" y2="280" class="solid"></line>
  <line x1="536" y1="280" x2="544" y2="280" class="solid"></line>
  <polygon points="519,332 526,324 526,335" class="filled"></polygon>
  <line x1="522" y1="332" x2="502" y2="372" class="solid"></line>
  <polygon points="550.8,348 550.8,340 545.2,340 545.2,348" class="filled"></polygon>
  <line x1="546" y1="348" x2="526" y2="388" class="solid"></line>
  <polygon points="519,364 526,356 526,367" class="filled"></polygon>
  <line x1="522" y1="364" x2="504" y2="400" class="solid"></line>
  <polygon points="502.8,380 502.8,372 497.2,372 497.2,380" class="filled"></polygon>
  <polygon points="529,388 522,396 522,385" class="filled"></polygon>
  <rect x="416" y="356" width="8" height="8" class="solid filled" rx="0"></rect>
  <line x1="420" y1="364" x2="420" y2="404" class="solid"></line>
  <polygon points="416,404 424,404 420,416" class="filled"></polygon>
  <line x1="360" y1="376" x2="348" y2="376" class="solid end_marked_open_circle"></line>
  <line x1="360" y1="376" x2="384" y2="376" class="solid"></line>
  <polygon points="384,372 392,376 384,380" class="filled"></polygon>
  <polygon points="352,388 344,392 352,396" class="filled"></polygon>
  <line x1="352" y1="392" x2="384" y2="392" class="solid"></line>
  <rect x="384" y="388" width="8" height="8" class="solid filled" rx="0"></rect>
  <polygon points="432,380 436,368 440,380" class="filled"></polygon>
  <line x1="436" y1="380" x2="436" y2="424" class="solid end_marked_open_circle"></line>
  <polygon points="456,396 460,384 464,396" class="filled"></polygon>
  <line x1="460" y1="396" x2="460" y2="432" class="broken"></line>
  <line x1="476" y1="384" x2="476" y2="420" class="broken"></line>
  <polygon points="472,420 480,420 476,432" class="filled"></polygon>
  <line x1="478" y1="460" x2="476" y2="456" class="solid end_marked_big_open_circle"></line>
  <line x1="480" y1="464" x2="498" y2="500" class="solid"></line>
  <polygon points="495,500 502,508 502,497" class="filled"></polygon>
  <polygon points="240,500 248,504 240,508" class="filled"></polygon>
  <polygon points="272,468 264,472 272,476" class="filled"></polygon>
  <polygon points="336,468 344,472 336,476" class="filled"></polygon>
  <polygon points="360,500 352,504 360,508" class="filled"></polygon>
  <polygon points="465,476 458,468 458,479" class="filled"></polygon>
  <line x1="462" y1="476" x2="484" y2="520" class="solid end_marked_circle"></line>
  <line x1="24" y1="888" x2="12" y2="888" class="solid end_marked_open_circle"></line>
  <polygon points="24,884 32,888 24,892" class="filled"></polygon>
  <text x="50" y="892" >Quick</text>
  <text x="98" y="892" >logo</text>
  <line x1="66" y1="908" x2="44" y2="952" class="solid end_marked_circle"></line>
  <line x1="60" y1="920" x2="76" y2="920" class="solid end_marked_open_circle"></line>
  <line x1="80" y1="920" x2="112" y2="920" class="solid"></line>
  <polygon points="96,932 104,936 96,940" class="filled"></polygon>
  <text x="138" y="892" >scribbles</text>
  <line x1="276" y1="944" x2="276" y2="968" class="solid end_marked_circle"></line>
  <line x1="172" y1="956" x2="172" y2="952" class="solid end_marked_big_open_circle"></line>
  <line x1="416" y1="952" x2="436" y2="952" class="solid end_marked_circle"></line>
  <line x1="432" y1="952" x2="448" y2="952" class="solid"></line>
  <polygon points="344,948 352,952 344,956" class="filled"></polygon>
  <text x="58" y="1100" >.::::.</text>
  <line x1="24" y1="1160" x2="12" y2="1160" class="solid end_marked_open_circle"></line>
  <polygon points="24,1156 32,1160 24,1164" class="filled"></polygon>
  <text x="50" y="1164" >Even</text>
  <text x="90" y="1164" >unicode</text>
  <text x="154" y="1164" >box</text>
  <text x="186" y="1164" >drawing</text>
  <text x="250" y="1164" >characters</text>
  <text x="338" y="1164" >are</text>
  <text x="370" y="1164" >supported</text>
  <line x1="24" y1="1288" x2="12" y2="1288" class="solid end_marked_open_circle"></line>
  <polygon points="24,1284 32,1288 24,1292" class="filled"></polygon>
  <text x="42" y="1292" >Circle,</text>
  <text x="106" y="1292" >quarter</text>
  <text x="170" y="1292" >arcs,</text>
  <text x="218" y="1292" >half</text>
  <text x="258" y="1292" >circles,</text>
  <text x="330" y="1292" >3</text>
  <line x1="344" y1="1280" x2="336" y2="1296" class="solid"></line>
  <text x="346" y="1292" >4</text>
  <text x="362" y="1292" >quarter</text>
  <text x="426" y="1292" >arcs</text>
  <line x1="24" y1="1560" x2="12" y2="1560" class="solid end_marked_open_circle"></line>
  <polygon points="24,1556 32,1560 24,1564" class="filled"></polygon>
  <text x="42" y="1564" >Grids</text>
  <text x="42" y="1660" >{r}</text>
  <text x="42" y="1724" >{g}</text>
  <line x1="24" y1="2024" x2="12" y2="2024" class="solid end_marked_open_circle"></line>
  <polygon points="24,2020 32,2024 24,2028" class="filled"></polygon>
  <text x="42" y="2028" >Graphics</text>
  <text x="114" y="2028" >Diagram</text>
  <line x1="618" y1="2060" x2="620" y2="2056" class="solid end_marked_circle"></line>
  <polygon points="569,2076 562,2068 562,2079" class="filled"></polygon>
  <line x1="566" y1="2076" x2="586" y2="2116" class="solid"></line>
  <line x1="616" y1="2064" x2="598" y2="2100" class="solid"></line>
  <polygon points="601,2100 594,2108 594,2097" class="filled"></polygon>
  <line x1="520" y1="2120" x2="588" y2="2120" class="solid end_marked_circle"></line>
  <line x1="584" y1="2120" x2="656" y2="2120" class="solid"></line>
  <text x="34" y="2076" >0</text>
  <text x="98" y="2076" >3</text>
  <line x1="44" y1="2088" x2="44" y2="2088" class="solid end_marked_circle"></line>
  <line x1="42" y1="2092" x2="28" y2="2120" class="solid end_marked_circle"></line>
  <line x1="44" y1="2092" x2="44" y2="2152" class="solid end_marked_circle"></line>
  <line x1="48" y1="2088" x2="108" y2="2088" class="solid end_marked_circle"></line>
  <line x1="104" y1="2088" x2="108" y2="2088" class="solid"></line>
  <line x1="106" y1="2092" x2="94" y2="2116" class="solid"></line>
  <line x1="108" y1="2092" x2="108" y2="2148" class="solid"></line>
  <text x="18" y="2108" >1</text>
  <text x="82" y="2108" >2</text>
  <line x1="28" y1="2120" x2="92" y2="2120" class="solid end_marked_circle"></line>
  <line x1="88" y1="2120" x2="92" y2="2120" class="solid"></line>
  <line x1="28" y1="2124" x2="28" y2="2180" class="solid"></line>
  <line x1="92" y1="2124" x2="92" y2="2180" class="solid"></line>
  <text x="50" y="2140" >4</text>
  <text x="114" y="2140" >7</text>
  <line x1="44" y1="2152" x2="108" y2="2152" class="solid end_marked_circle"></line>
  <line x1="104" y1="2152" x2="108" y2="2152" class="solid"></line>
  <line x1="42" y1="2156" x2="28" y2="2184" class="solid end_marked_circle"></line>
  <line x1="106" y1="2156" x2="94" y2="2180" class="solid"></line>
  <line x1="28" y1="2184" x2="92" y2="2184" class="solid end_marked_circle"></line>
  <line x1="88" y1="2184" x2="92" y2="2184" class="solid"></line>
  <text x="18" y="2204" >5</text>
  <text x="82" y="2204" >6</text>
  <text x="314" y="2076" >P</text>
  <line x1="334" y1="2076" x2="332" y2="2072" class="solid end_marked_circle"></line>
  <line x1="336" y1="2080" x2="370" y2="2148" class="solid"></line>
  <text x="306" y="2124" >v0</text>
  <text x="418" y="2124" >v3</text>
  <line x1="324" y1="2136" x2="324" y2="2136" class="solid end_marked_circle"></line>
  <line x1="322" y1="2140" x2="292" y2="2200" class="solid end_marked_circle"></line>
  <line x1="328" y1="2136" x2="412" y2="2136" class="solid end_marked_circle"></line>
  <line x1="408" y1="2136" x2="412" y2="2136" class="solid"></line>
  <line x1="414" y1="2140" x2="442" y2="2196" class="solid"></line>
  <polygon points="367,2148 374,2156 374,2145" class="filled"></polygon>
  <text x="386" y="2156" >X</text>
  <line x1="384" y1="2160" x2="380" y2="2168" class="solid end_marked_open_circle"></line>
  <line x1="292" y1="2200" x2="444" y2="2200" class="solid end_marked_circle"></line>
  <line x1="440" y1="2200" x2="444" y2="2200" class="solid"></line>
  <text x="450" y="2076" >Eye</text>
  <path d="M 480,2080 A 16,16 0,0,1 480,2096" class="nofill"></path>
  <text x="162" y="2092" >+y</text>
  <polygon points="168,2108 172,2096 176,2108" class="filled"></polygon>
  <polygon points="192,2132 184,2136 192,2140" class="filled"></polygon>
  <path d="M 160,2148 A 4,4 0,0,0 156,2156" class="nofill"></path>
  <polygon points="159,2159 157,2155 155,2157" class="filled"></polygon>
  <polygon points="216,2148 224,2152 216,2156" class="filled"></polygon>
  <text x="178" y="2172" >⤴</text>
  <polygon points="161,2180 154,2188 154,2177" class="filled"></polygon>
  <text x="138" y="2204" >+z</text>
  <text x="626" y="2092" >Reflection</text>
  <text x="234" y="2156" >+x</text>
  <polygon points="536,2148 528,2152 536,2156" class="filled"></polygon>
  <line x1="536" y1="2152" x2="600" y2="2152" class="solid"></line>
  <text x="546" y="2172" >Refraction</text>
  <line x1="508" y1="2192" x2="508" y2="2212" class="solid"></line>
  <polygon points="504,2212 512,2212 508,2224" class="filled"></polygon>
  <line x1="666" y1="2156" x2="668" y2="2152" class="solid end_marked_open_circle"></line>
  <line x1="670" y1="2156" x2="690" y2="2196" class="solid"></line>
  <line x1="664" y1="2160" x2="644" y2="2200" class="solid end_marked_open_circle"></line>
  <line x1="648" y1="2200" x2="692" y2="2200" class="solid end_marked_open_circle"></line>
  <text x="266" y="2204" >v1</text>
  <text x="458" y="2204" >v2</text>
  <line x1="24" y1="2264" x2="12" y2="2264" class="solid end_marked_open_circle"></line>
  <polygon points="24,2260 32,2264 24,2268" class="filled"></polygon>
  <text x="42" y="2268" >CJK</text>
  <text x="74" y="2268" >characters</text>
  <line x1="212" y1="2288" x2="212" y2="2336" class="solid"></line>
  <line x1="340" y1="2288" x2="340" y2="2336" class="solid"></line>
  <line x1="24" y1="2360" x2="12" y2="2360" class="solid end_marked_open_circle"></line>
  <polygon points="24,2356 32,2360 24,2364" class="filled"></polygon>
  <text x="50" y="2364" >Sequence</text>
  <text x="122" y="2364" >Diagrams</text>
  <polygon points="312,2388 320,2392 312,2396" class="filled"></polygon>
  <text x="82" y="2412" >A</text>
  <text x="146" y="2412" >B</text>
  <text x="202" y="2412" >C</text>
  <text x="226" y="2412" >D</text>
  <line x1="84" y1="2424" x2="84" y2="2424" class="solid end_marked_circle"></line>
  <line x1="88" y1="2424" x2="148" y2="2424" class="solid end_marked_circle"></line>
  <line x1="144" y1="2424" x2="196" y2="2424" class="solid end_marked_circle"></line>
  <line x1="150" y1="2428" x2="162" y2="2452" class="solid"></line>
  <line x1="192" y1="2424" x2="228" y2="2424" class="solid end_marked_circle"></line>
  <line x1="224" y1="2424" x2="268" y2="2424" class="solid end_marked_circle"></line>
  <line x1="264" y1="2424" x2="312" y2="2424" class="solid"></line>
  <polygon points="312,2420 320,2424 312,2428" class="filled"></polygon>
  <polygon points="255,2444 262,2436 262,2447" class="filled"></polygon>
  <polygon points="159,2452 166,2460 166,2449" class="filled"></polygon>
  <polygon points="312,2452 320,2456 312,2460" class="filled"></polygon>
  <text x="170" y="2476" >B</text>
  <text x="338" y="2396" >F</text>
  <text x="338" y="2428" >E</text>
  <text x="338" y="2460" >G</text>
  <line x1="184" y1="2472" x2="200" y2="2472" class="solid"></line>
  <polygon points="200,2468 208,2472 200,2476" class="filled"></polygon>
  <text x="218" y="2476" >C</text>
  <line x1="216" y1="2544" x2="208" y2="2560" class="solid"></line>
  <line x1="224" y1="2544" x2="232" y2="2560" class="solid"></line>
  <text x="82" y="2588" >Bob</text>
  <text x="202" y="2604" >Alice</text>
  <text x="130" y="2620" >hello</text>
  <line x1="220" y1="2608" x2="220" y2="2688" class="solid"></line>
  <polygon points="208,2628 216,2632 208,2636" class="filled"></polygon>
  <text x="114" y="2668" >Is</text>
  <polygon points="104,2676 96,2680 104,2684" class="filled"></polygon>
  <line x1="104" y1="2680" x2="112" y2="2680" class="solid"></line>
  <line x1="120" y1="2680" x2="128" y2="2680" class="solid"></line>
  <line x1="200" y1="2680" x2="208" y2="2680" class="solid"></line>
  <text x="202" y="2700" >Alice</text>
  <text x="82" y="2716" >Bob</text>
  <line x1="216" y1="2736" x2="208" y2="2752" class="solid"></line>
  <line x1="224" y1="2736" x2="232" y2="2752" class="solid"></line>
  <text x="138" y="2668" >it</text>
  <text x="162" y="2668" >ok?</text>
  <line x1="136" y1="2680" x2="144" y2="2680" class="solid"></line>
  <line x1="152" y1="2680" x2="160" y2="2680" class="solid"></line>
  <line x1="168" y1="2680" x2="176" y2="2680" class="solid"></line>
  <line x1="184" y1="2680" x2="192" y2="2680" class="solid"></line>
  <text x="122" y="2844" >0</text>
  <line x1="104" y1="2864" x2="86" y2="2900" class="solid"></line>
  <polygon points="89,2900 82,2908 82,2897" class="filled"></polygon>
  <text x="82" y="2940" >1</text>
  <line x1="88" y1="2960" x2="106" y2="2996" class="solid"></line>
  <polygon points="103,2996 110,3004 110,2993" class="filled"></polygon>
  <text x="114" y="3036" >3</text>
  <line x1="144" y1="2864" x2="162" y2="2900" class="solid"></line>
  <polygon points="159,2900 166,2908 166,2897" class="filled"></polygon>
  <text x="178" y="2940" >4</text>
  <text x="202" y="2956" >.</text>
  <line x1="172" y1="2960" x2="172" y2="3012" class="solid"></line>
  <line x1="192" y1="2960" x2="218" y2="3012" class="solid"></line>
  <line x1="204" y1="2952" x2="252" y2="3000" class="broken"></line>
  <text x="250" y="3004" >.</text>
  <polygon points="168,3012 176,3012 172,3024" class="filled"></polygon>
  <polygon points="215,3012 222,3020 222,3009" class="filled"></polygon>
  <polygon points="256,3012 264,3016 260,3008" class="filled"></polygon>
  <text x="178" y="3052" >5</text>
  <text x="226" y="3052" >6</text>
  <text x="274" y="3052" >7</text>
  <line x1="64" y1="2960" x2="46" y2="2996" class="solid"></line>
  <polygon points="49,2996 42,3004 42,2993" class="filled"></polygon>
  <text x="42" y="3036" >2</text>
  <line x1="24" y1="3128" x2="12" y2="3128" class="solid end_marked_open_circle"></line>
  <polygon points="24,3124 32,3128 24,3132" class="filled"></polygon>
  <text x="42" y="3132" >Plot</text>
  <text x="82" y="3132" >diagrams</text>
  <polygon points="64,3164 68,3152 72,3164" class="filled"></polygon>
  <line x1="68" y1="3164" x2="68" y2="3224" class="broken end_marked_circle"></line>
  <polygon points="296,3220 304,3224 296,3228" class="filled"></polygon>
  <text x="34" y="3180" >Uin</text>
  <text x="42" y="3260" >Udc</text>
  <polygon points="64,3260 68,3248 72,3260" class="filled"></polygon>
  <line x1="68" y1="3260" x2="68" y2="3320" class="broken end_marked_circle"></line>
  <line x1="140" y1="3280" x2="140" y2="3344" class="broken"></line>
  <line x1="144" y1="3320" x2="296" y2="3320" class="broken"></line>
  <polygon points="296,3316 304,3320 296,3324" class="filled"></polygon>
  <polygon points="152,3332 144,3336 152,3340" class="filled"></polygon>
  <line x1="152" y1="3336" x2="192" y2="3336" class="solid"></line>
  <polygon points="192,3332 200,3336 192,3340" class="filled"></polygon>
  <line x1="204" y1="3328" x2="204" y2="3376" class="broken"></line>
  <polygon points="64,3356 68,3344 72,3356" class="filled"></polygon>
  <line x1="68" y1="3356" x2="68" y2="3432" class="broken end_marked_circle"></line>
  <text x="154" y="3356" >500ms</text>
  <text x="226" y="3404" >Inactive</text>
  <text x="106" y="3420" >Active</text>
  <polygon points="296,3428 304,3432 296,3436" class="filled"></polygon>
  <text x="2" y="3388" >Cpu.Qon</text>
  <line x1="24" y1="3464" x2="12" y2="3464" class="solid end_marked_open_circle"></line>
  <polygon points="24,3460 32,3464 24,3468" class="filled"></polygon>
  <text x="42" y="3468" >Railroad</text>
  <text x="114" y="3468" >diagrams</text>
  <line x1="80" y1="3512" x2="68" y2="3512" class="solid end_marked_open_circle"></line>
  <text x="138" y="3516" >elem</text>
  <text x="218" y="3516" >;</text>
  <text x="266" y="3516" >n</text>
  <line x1="312" y1="3512" x2="332" y2="3512" class="solid end_marked_open_circle"></line>
  <polygon points="160,3540 168,3544 160,3548" class="filled"></polygon>
  <line x1="112" y1="3576" x2="120" y2="3576" class="solid"></line>
  <text x="162" y="3580" >x</text>
  <line x1="208" y1="3576" x2="216" y2="3576" class="solid"></line>
  <line x1="308" y1="3584" x2="308" y2="3616" class="solid"></line>
  <text x="162" y="3628" >,</text>
  <polygon points="304,3628 308,3616 312,3628" class="filled"></polygon>
  <polygon points="184,3652 192,3656 184,3660" class="filled"></polygon>
  <text x="162" y="3692" >x</text>
  <text x="218" y="3692" >,</text>
  <polygon points="200,3716 192,3720 200,3724" class="filled"></polygon>
  <polygon points="416,3732 424,3736 416,3740" class="filled"></polygon>
  <text x="50" y="3772" >O</text>
  <text x="106" y="3772" >struct</text>
  <line x1="152" y1="3776" x2="160" y2="3776" class="solid"></line>
  <text x="162" y="3772" >name</text>
  <text x="234" y="3772" >:</text>
  <line x1="256" y1="3776" x2="264" y2="3776" class="solid"></line>
  <line x1="284" y1="3760" x2="284" y2="3776" class="solid"></line>
  <text x="354" y="3772" >name</text>
  <text x="426" y="3772" >:</text>
  <text x="482" y="3772" >tpe</text>
  <line x1="596" y1="3760" x2="596" y2="3776" class="solid"></line>
  <text x="650" y="3772" >body</text>
  <text x="746" y="3772" >O</text>
  <polygon points="432,3812 424,3816 432,3820" class="filled"></polygon>
  <line x1="432" y1="3816" x2="480" y2="3816" class="solid"></line>
  <text x="498" y="3820" >,</text>
  <line x1="24" y1="3880" x2="12" y2="3880" class="solid end_marked_open_circle"></line>
  <polygon points="24,3876 32,3880 24,3884" class="filled"></polygon>
  <text x="42" y="3884" >Statistical</text>
  <text x="138" y="3884" >charts</text>
  <text x="34" y="3932" >E</text>
  <text x="34" y="3948" >D</text>
  <text x="34" y="3964" >C</text>
  <text x="34" y="3980" >B</text>
  <text x="34" y="3996" >A</text>
  <line x1="52" y1="3928" x2="260" y2="3928" class="solid end_marked_circle"></line>
  <line x1="52" y1="3928" x2="52" y2="3992" class="solid"></line>
  <line x1="52" y1="3944" x2="212" y2="3944" class="solid end_marked_circle"></line>
  <line x1="208" y1="3944" x2="236" y2="3944" class="solid end_marked_circle"></line>
  <line x1="232" y1="3944" x2="284" y2="3944" class="solid end_marked_circle"></line>
  <line x1="280" y1="3944" x2="284" y2="3944" class="solid"></line>
  <line x1="52" y1="3960" x2="164" y2="3960" class="solid end_marked_circle"></line>
  <line x1="160" y1="3960" x2="188" y2="3960" class="solid end_marked_circle"></line>
  <line x1="184" y1="3960" x2="188" y2="3960" class="solid"></line>
  <line x1="52" y1="3976" x2="116" y2="3976" class="solid end_marked_circle"></line>
  <line x1="112" y1="3976" x2="140" y2="3976" class="solid end_marked_circle"></line>
  <line x1="136" y1="3976" x2="140" y2="3976" class="solid"></line>
  <line x1="52" y1="3992" x2="68" y2="3992" class="solid end_marked_circle"></line>
  <line x1="64" y1="3992" x2="92" y2="3992" class="solid end_marked_circle"></line>
  <text x="66" y="4012" >5</text>
  <text x="82" y="4012" >10</text>
  <text x="106" y="4012" >15</text>
  <text x="130" y="4012" >20</text>
  <text x="154" y="4012" >25</text>
  <text x="178" y="4012" >30</text>
  <text x="202" y="4012" >35</text>
  <text x="226" y="4012" >40</text>
  <text x="250" y="4012" >45</text>
  <text x="274" y="4012" >50</text>
  <text x="330" y="3932" >E</text>
  <text x="330" y="3948" >D</text>
  <text x="330" y="3964" >C</text>
  <text x="330" y="3980" >B</text>
  <text x="330" y="3996" >A</text>
  <line x1="348" y1="3920" x2="348" y2="3992" class="solid"></line>
  <line x1="556" y1="3932" x2="556" y2="3928" class="solid end_marked_open_circle"></line>
  <line x1="508" y1="3948" x2="508" y2="3944" class="solid end_marked_open_circle"></line>
  <line x1="532" y1="3948" x2="532" y2="3944" class="solid end_marked_open_circle"></line>
  <line x1="556" y1="3936" x2="556" y2="4000" class="solid"></line>
  <line x1="580" y1="3948" x2="580" y2="3944" class="solid end_marked_open_circle"></line>
  <line x1="460" y1="3964" x2="460" y2="3960" class="solid end_marked_open_circle"></line>
  <line x1="484" y1="3964" x2="484" y2="3960" class="solid end_marked_open_circle"></line>
  <line x1="508" y1="3952" x2="508" y2="4000" class="solid"></line>
  <line x1="532" y1="3952" x2="532" y2="4000" class="solid"></line>
  <line x1="412" y1="3980" x2="412" y2="3976" class="solid end_marked_open_circle"></line>
  <line x1="436" y1="3980" x2="436" y2="3976" class="solid end_marked_open_circle"></line>
  <line x1="460" y1="3968" x2="460" y2="4000" class="solid"></line>
  <line x1="484" y1="3968" x2="484" y2="4000" class="solid"></line>
  <line x1="348" y1="3992" x2="364" y2="3992" class="solid end_marked_open_circle"></line>
  <line x1="368" y1="3992" x2="388" y2="3992" class="solid end_marked_open_circle"></line>
  <line x1="412" y1="3984" x2="412" y2="4000" class="solid"></line>
  <line x1="436" y1="3984" x2="436" y2="4000" class="solid"></line>
  <text x="362" y="4012" >5</text>
  <text x="378" y="4012" >10</text>
  <text x="402" y="4012" >15</text>
  <text x="426" y="4012" >20</text>
  <text x="450" y="4012" >25</text>
  <text x="474" y="4012" >30</text>
  <text x="498" y="4012" >35</text>
  <text x="522" y="4012" >40</text>
  <text x="546" y="4012" >45</text>
  <text x="570" y="4012" >50</text>
  <text x="18" y="4076" >85.67</text>
  <text x="18" y="4092" >78.20</text>
  <text x="18" y="4108" >70.73</text>
  <text x="18" y="4124" >63.27</text>
  <text x="18" y="4140" >55.80</text>
  <text x="18" y="4156" >48.33</text>
  <text x="18" y="4172" >40.87</text>
  <text x="18" y="4188" >33.40</text>
  <text x="18" y="4204" >25.93</text>
  <text x="18" y="4220" >18.47</text>
  <text x="18" y="4236" >11.00</text>
  <text x="50" y="4268" >2011</text>
  <text x="146" y="4268" >2012</text>
  <text x="242" y="4268" >2013</text>
  <text x="338" y="4268" >2014</text>
  <text x="434" y="4268" >2015</text>
  <text x="530" y="4268" >2016</text>
  <line x1="24" y1="4392" x2="12" y2="4392" class="solid end_marked_open_circle"></line>
  <polygon points="24,4388 32,4392 24,4396" class="filled"></polygon>
  <text x="50" y="4396" >Flow</text>
  <text x="90" y="4396" >charts</text>
  <polygon points="176,4436 184,4436 180,4448" class="filled"></polygon>
  <polygon points="96,4468 104,4472 96,4476" class="filled"></polygon>
  <line x1="120" y1="4480" x2="130" y2="4500" class="solid"></line>
  <line x1="176" y1="4480" x2="166" y2="4500" class="solid"></line>
  <polygon points="127,4500 134,4508 134,4497" class="filled"></polygon>
  <polygon points="169,4500 162,4508 162,4497" class="filled"></polygon>
  <polygon points="168,4564 176,4568 168,4572" class="filled"></polygon>
  <line x1="324" y1="4432" x2="324" y2="4452" class="solid"></line>
  <polygon points="320,4452 328,4452 324,4464" class="filled"></polygon>
  <line x1="380" y1="4432" x2="380" y2="4452" class="solid"></line>
  <polygon points="376,4452 384,4452 380,4464" class="filled"></polygon>
  <polygon points="392,4564 384,4568 392,4572" class="filled"></polygon>
  <line x1="484" y1="4432" x2="484" y2="4452" class="solid"></line>
  <polygon points="480,4452 488,4452 484,4464" class="filled"></polygon>
  <line x1="556" y1="4432" x2="556" y2="4532" class="solid"></line>
  <polygon points="552,4532 560,4532 556,4544" class="filled"></polygon>
  <line x1="364" y1="4512" x2="364" y2="4532" class="solid"></line>
  <polygon points="360,4532 368,4532 364,4544" class="filled"></polygon>
  <line x1="484" y1="4512" x2="484" y2="4612" class="solid"></line>
  <polygon points="480,4612 488,4612 484,4624" class="filled"></polygon>
  <line x1="364" y1="4592" x2="364" y2="4612" class="solid"></line>
  <polygon points="360,4612 368,4612 364,4624" class="filled"></polygon>
  <line x1="564" y1="4592" x2="564" y2="4612" class="solid"></line>
  <polygon points="560,4612 568,4612 564,4624" class="filled"></polygon>
  <line x1="612" y1="4432" x2="612" y2="4452" class="solid"></line>
  <polygon points="608,4452 616,4452 612,4464" class="filled"></polygon>
  <text x="234" y="4428" >OS</text>
  <text x="258" y="4428" >API</text>
  <line x1="24" y1="4712" x2="12" y2="4712" class="solid end_marked_open_circle"></line>
  <polygon points="24,4708 32,4712 24,4716" class="filled"></polygon>
  <text x="50" y="4716" >Block</text>
  <text x="98" y="4716" >diagrams</text>
  <text x="50" y="4764" >vncviewer</text>
  <line x1="56" y1="4784" x2="72" y2="4784" class="solid"></line>
  <text x="50" y="4796" >[</text>
  <line x1="56" y1="4800" x2="72" y2="4800" class="solid"></line>
  <text x="74" y="4796" >]</text>
  <line x1="88" y1="4790" x2="96" y2="4790" class="solid"></line>
  <line x1="88" y1="4794" x2="96" y2="4794" class="solid"></line>
  <line x1="48" y1="4800" x2="40" y2="4816" class="solid"></line>
  <text x="210" y="4764" >,</text>
  <path d="M 224,4752 A 16,16 0,0,0 224,4768" class="nofill"></path>
  <line x1="120" y1="4792" x2="152" y2="4792" class="solid"></line>
  <polygon points="152,4788 160,4792 152,4796" class="filled"></polygon>
  <path d="M 192,4800 A 16,16 0,0,0 192,4816" class="nofill"></path>
  <path d="M 240,4752 A 16,16 0,0,1 240,4768" class="nofill"></path>
  <text x="250" y="4764" >,</text>
  <text x="386" y="4780" >gateway</text>
  <text x="530" y="4780" >vncserver</text>
  <text x="298" y="4796" >)</text>
  <line x1="304" y1="4792" x2="360" y2="4792" class="solid"></line>
  <polygon points="360,4788 368,4792 360,4796" class="filled"></polygon>
  <line x1="376" y1="4800" x2="456" y2="4800" class="solid"></line>
  <line x1="464" y1="4792" x2="512" y2="4792" class="solid"></line>
  <polygon points="512,4788 520,4792 512,4796" class="filled"></polygon>
  <text x="282" y="4812" >.</text>
  <line x1="288" y1="4808" x2="296" y2="4808" class="solid"></line>
  <text x="298" y="4812" >&#39;</text>
  <text x="370" y="4812" >[</text>
  <line x1="376" y1="4816" x2="384" y2="4816" class="solid"></line>
  <text x="386" y="4812" >...</text>
  <line x1="408" y1="4816" x2="424" y2="4816" class="solid"></line>
  <path d="M 240,4816 A 16,16 0,0,1 240,4832" class="nofill"></path>
  <text x="250" y="4828" >.</text>
  <line x1="568" y1="4832" x2="560" y2="4848" class="solid"></line>
  <text x="202" y="4796" >internet</text>
  <text x="202" y="4828" >&#39;</text>
  <line x1="208" y1="4824" x2="216" y2="4824" class="solid"></line>
  <text x="218" y="4828" >.</text>
  <path d="M 232,4816 A 16,16 0,0,0 232,4832" class="nofill"></path>
  <text x="242" y="4892" >Valveless</text>
  <text x="226" y="4908" >Pulsejet</text>
  <text x="298" y="4908" >engine</text>
  <polygon points="377,4916 370,4924 370,4913" class="filled"></polygon>
  <line x1="328" y1="4984" x2="420" y2="4984" class="solid end_marked_open_circle"></line>
  <line x1="420" y1="4988" x2="420" y2="5192" class="solid end_marked_circle"></line>
  <text x="338" y="5004" >&#39;</text>
  <polygon points="264,5020 268,5008 272,5020" class="filled"></polygon>
  <text x="338" y="5020" >.</text>
  <text x="426" y="5052" >GND</text>
  <line x1="188" y1="5176" x2="268" y2="5176" class="solid end_marked_open_circle"></line>
  <line x1="188" y1="5176" x2="188" y2="5216" class="solid"></line>
  <line x1="204" y1="5176" x2="204" y2="5216" class="solid"></line>
  <line x1="220" y1="5176" x2="220" y2="5216" class="solid"></line>
  <line x1="296" y1="5176" x2="372" y2="5176" class="solid end_marked_circle"></line>
  <line x1="368" y1="5176" x2="540" y2="5176" class="solid end_marked_big_open_circle"></line>
  <line x1="372" y1="5180" x2="372" y2="5220" class="solid"></line>
  <text x="266" y="5196" >power</text>
  <line x1="420" y1="5192" x2="540" y2="5192" class="solid end_marked_big_open_circle"></line>
  <line x1="420" y1="5196" x2="420" y2="5220" class="solid"></line>
  <text x="266" y="5212" >switch</text>
  <line x1="348" y1="5224" x2="372" y2="5224" class="solid end_marked_open_circle"></line>
  <line x1="376" y1="5224" x2="420" y2="5224" class="solid end_marked_open_circle"></line>
  <text x="186" y="5244" >HHO</text>
  <text x="178" y="5260" >Generator</text>
  <text x="370" y="5260" >+</text>
  <line x1="416" y1="5256" x2="424" y2="5256" class="solid"></line>
  <text x="370" y="5292" >Battery</text>
  <line x1="488" y1="4952" x2="552" y2="4952" class="solid"></line>
  <polygon points="552,4948 560,4952 552,4956" class="filled"></polygon>
  <text x="498" y="4972" >thrust</text>
  <line x1="544" y1="4968" x2="560" y2="4968" class="solid"></line>
  <polygon points="560,4964 568,4968 560,4972" class="filled"></polygon>
  <line x1="496" y1="4984" x2="560" y2="4984" class="solid"></line>
  <polygon points="560,4980 568,4984 560,4988" class="filled"></polygon>
  <text x="154" y="5052" >fuel</text>
  <text x="146" y="5068" >intake</text>
  <line x1="200" y1="5056" x2="216" y2="5056" class="solid"></line>
  <text x="218" y="5052" >^</text>
  <text x="282" y="5052" >^</text>
  <line x1="288" y1="5056" x2="304" y2="5056" class="solid"></line>
  <text x="306" y="5052" >spark</text>
  <text x="306" y="5068" >plug</text>
  <polygon points="120,5188 128,5192 120,5196" class="filled"></polygon>
  <text x="82" y="5228" >Water</text>
  <text x="82" y="5244" >intake</text>
  <text x="546" y="5260" >Solar</text>
  <text x="594" y="5260" >panel</text>
  <line x1="120" y1="5318" x2="176" y2="5318" class="solid"></line>
  <line x1="120" y1="5322" x2="176" y2="5322" class="solid"></line>
  <line x1="128" y1="5334" x2="168" y2="5334" class="solid"></line>
  <line x1="128" y1="5338" x2="168" y2="5338" class="solid"></line>
  <line x1="136" y1="5350" x2="160" y2="5350" class="solid"></line>
  <line x1="136" y1="5354" x2="160" y2="5354" class="solid"></line>
  <line x1="144" y1="5366" x2="152" y2="5366" class="solid"></line>
  <line x1="144" y1="5370" x2="152" y2="5370" class="solid"></line>
  <line x1="148" y1="5376" x2="148" y2="5520" class="solid"></line>
  <text x="234" y="5404" >micro</text>
  <text x="282" y="5404" >henry</text>
  <text x="234" y="5420" >coil</text>
  <text x="274" y="5420" >w</text>
  <line x1="288" y1="5408" x2="280" y2="5424" class="solid"></line>
  <text x="290" y="5420" >tuning</text>
  <text x="202" y="5452" >&#39;</text>
  <text x="282" y="5484" >pico</text>
  <text x="322" y="5484" >farad</text>
  <text x="370" y="5484" >cap</text>
  <path d="M 312,5488 A 16,16 0,0,0 312,5504" class="nofill"></path>
  <text x="314" y="5500" >trimmable</text>
  <path d="M 384,5488 A 16,16 0,0,1 384,5504" class="nofill"></path>
  <text x="138" y="5564" >ground</text>
  <text x="194" y="5564" >plane</text>
  <path d="M 248,5552 A 16,16 0,0,0 248,5568" class="nofill"></path>
  <text x="250" y="5564" >foil</text>
  <path d="M 280,5552 A 16,16 0,0,1 280,5568" class="nofill"></path>
  <text x="186" y="5340" >symbolic</text>
  <text x="258" y="5340" >antenna</text>
  <text x="346" y="5420" >lug</text>
  <text x="66" y="5532" >PC</text>
  <line x1="104" y1="5528" x2="112" y2="5528" class="solid"></line>
  <polygon points="112,5524 120,5528 112,5532" class="filled"></polygon>
  <text x="66" y="5548" >Board</text>
  <line x1="24" y1="5640" x2="12" y2="5640" class="solid end_marked_open_circle"></line>
  <polygon points="24,5636 32,5640 24,5644" class="filled"></polygon>
  <text x="42" y="5644" >Mindmaps</text>
  <line x1="354" y1="5676" x2="284" y2="5816" class="solid end_marked_open_circle"></line>
  <polygon points="376,5668 384,5672 376,5676" class="filled"></polygon>
  <polygon points="376,5700 384,5704 376,5708" class="filled"></polygon>
  <line x1="156" y1="5720" x2="156" y2="5720" class="solid end_marked_circle"></line>
  <line x1="222" y1="5724" x2="268" y2="5816" class="solid end_marked_open_circle"></line>
  <line x1="300" y1="5784" x2="376" y2="5784" class="solid"></line>
  <polygon points="376,5780 384,5784 376,5788" class="filled"></polygon>
  <line x1="300" y1="5832" x2="300" y2="5832" class="solid end_marked_circle"></line>
  <line x1="360" y1="5848" x2="412" y2="5848" class="solid end_marked_big_open_circle"></line>
  <line x1="416" y1="5848" x2="424" y2="5848" class="solid"></line>
  <polygon points="424,5844 432,5848 424,5852" class="filled"></polygon>
  <line x1="302" y1="5868" x2="300" y2="5864" class="solid end_marked_open_circle"></line>
  <line x1="250" y1="5884" x2="252" y2="5880" class="solid end_marked_open_circle"></line>
  <line x1="278" y1="5884" x2="276" y2="5880" class="solid end_marked_open_circle"></line>
  <line x1="294" y1="5884" x2="292" y2="5880" class="solid end_marked_open_circle"></line>
  <polygon points="384,5924 392,5928 384,5932" class="filled"></polygon>
  <line x1="192" y1="5952" x2="174" y2="5988" class="solid"></line>
  <polygon points="424,5956 432,5960 424,5964" class="filled"></polygon>
  <polygon points="177,5988 170,5996 170,5985" class="filled"></polygon>
  <polygon points="392,5988 400,5992 392,5996" class="filled"></polygon>
  <polygon points="177,6020 170,6028 170,6017" class="filled"></polygon>
  <text x="106" y="6044" >Worklaod</text>
  <polygon points="376,6036 384,6040 376,6044" class="filled"></polygon>
  <polygon points="384,6036 392,6040 384,6044" class="filled"></polygon>
  <polygon points="177,6052 170,6060 170,6049" class="filled"></polygon>
  <polygon points="177,6084 170,6092 170,6081" class="filled"></polygon>
  <text x="402" y="5676" >Alpha</text>
  <text x="402" y="5708" >Initial</text>
  <polygon points="448,5732 456,5736 448,5740" class="filled"></polygon>
  <polygon points="456,5764 464,5768 456,5772" class="filled"></polygon>
  <text x="466" y="5708" >Release</text>
  <text x="82" y="5724" >Planning</text>
  <text x="466" y="5740" >Patch</text>
  <text x="514" y="5740" >1</text>
  <text x="50" y="5756" >Initial</text>
  <text x="114" y="5756" >research</text>
  <line x1="134" y1="5772" x2="132" y2="5768" class="solid end_marked_circle"></line>
  <polygon points="224,5844 232,5848 224,5852" class="filled"></polygon>
  <text x="482" y="5772" >Patch</text>
  <text x="530" y="5772" >2</text>
  <text x="394" y="5788" >Beta</text>
  <text x="450" y="5852" >.</text>
  <text x="466" y="5852" >Release</text>
  <text x="530" y="5852" >.</text>
  <text x="410" y="5932" >Push</text>
  <text x="450" y="5932" >backs</text>
  <text x="442" y="5964" >Setbacks</text>
  <text x="410" y="5996" >Reception</text>
  <text x="130" y="6012" >Team</text>
  <text x="402" y="6044" >Career</text>
  <text x="458" y="6044" >change</text>
  <text x="138" y="6076" >PTO</text>
  <text x="138" y="6108" >Bug</text>
  <line x1="24" y1="6152" x2="12" y2="6152" class="solid end_marked_open_circle"></line>
  <polygon points="24,6148 32,6152 24,6156" class="filled"></polygon>
  <text x="50" y="6156" >It</text>
  <text x="74" y="6156" >can</text>
  <text x="106" y="6156" >do</text>
  <text x="130" y="6156" >complex</text>
  <text x="194" y="6156" >stuff</text>
  <text x="242" y="6156" >such</text>
  <text x="282" y="6156" >as</text>
  <text x="306" y="6156" >circuit</text>
  <text x="370" y="6156" >diagrams</text>
  <text x="58" y="6204" >+10</text>
  <line x1="80" y1="6200" x2="88" y2="6200" class="solid"></line>
  <text x="90" y="6204" >15V</text>
  <text x="226" y="6204" >0,047R</text>
  <line x1="52" y1="6216" x2="52" y2="6216" class="solid end_marked_circle"></line>
  <line x1="52" y1="6220" x2="52" y2="6240" class="solid"></line>
  <line x1="56" y1="6216" x2="132" y2="6216" class="solid end_marked_open_circle"></line>
  <line x1="136" y1="6216" x2="180" y2="6216" class="solid end_marked_open_circle"></line>
  <line x1="228" y1="6216" x2="244" y2="6216" class="solid end_marked_open_circle"></line>
  <line x1="248" y1="6216" x2="268" y2="6216" class="solid end_marked_open_circle"></line>
  <line x1="272" y1="6216" x2="348" y2="6216" class="solid end_marked_open_circle"></line>
  <line x1="352" y1="6216" x2="388" y2="6216" class="solid end_marked_open_circle"></line>
  <text x="34" y="6236" >+</text>
  <line x1="32" y1="6248" x2="40" y2="6248" class="solid"></line>
  <line x1="40" y1="6246" x2="64" y2="6246" class="solid"></line>
  <line x1="40" y1="6250" x2="64" y2="6250" class="solid"></line>
  <line x1="64" y1="6248" x2="72" y2="6248" class="solid"></line>
  <line x1="32" y1="6264" x2="40" y2="6264" class="solid"></line>
  <line x1="40" y1="6262" x2="64" y2="6262" class="solid"></line>
  <line x1="40" y1="6266" x2="64" y2="6266" class="solid"></line>
  <line x1="64" y1="6264" x2="72" y2="6264" class="solid"></line>
  <text x="122" y="6268" >.</text>
  <line x1="32" y1="6280" x2="40" y2="6280" class="solid"></line>
  <line x1="40" y1="6278" x2="64" y2="6278" class="solid"></line>
  <line x1="40" y1="6282" x2="64" y2="6282" class="solid"></line>
  <line x1="64" y1="6280" x2="72" y2="6280" class="solid"></line>
  <text x="106" y="6284" >470</text>
  <line x1="132" y1="6272" x2="132" y2="6308" class="solid"></line>
  <text x="146" y="6284" >+</text>
  <line x1="32" y1="6296" x2="40" y2="6296" class="solid"></line>
  <line x1="52" y1="6288" x2="52" y2="6312" class="solid"></line>
  <text x="114" y="6300" >uF</text>
  <line x1="52" y1="6312" x2="132" y2="6312" class="solid end_marked_open_circle"></line>
  <text x="210" y="6316" >6</text>
  <text x="250" y="6316" >7</text>
  <text x="274" y="6316" >8</text>
  <line x1="112" y1="6344" x2="152" y2="6344" class="solid"></line>
  <line x1="120" y1="6348" x2="144" y2="6348" class="solid"></line>
  <line x1="128" y1="6360" x2="136" y2="6360" class="solid"></line>
  <text x="298" y="6364" >1</text>
  <text x="122" y="6380" >GND</text>
  <line x1="292" y1="6376" x2="348" y2="6376" class="solid end_marked_open_circle"></line>
  <text x="378" y="6396" >`</text>
  <polygon points="384,6388 392,6392 384,6396" class="filled"></polygon>
  <line x1="388" y1="6400" x2="388" y2="6424" class="solid end_marked_open_circle"></line>
  <text x="306" y="6428" >220R</text>
  <polygon points="448,6436 456,6440 448,6444" class="filled"></polygon>
  <line x1="452" y1="6456" x2="452" y2="6488" class="solid end_marked_open_circle"></line>
  <text x="474" y="6476" >BYV29</text>
  <line x1="552" y1="6472" x2="560" y2="6472" class="solid"></line>
  <text x="562" y="6476" >12V6</text>
  <line x1="452" y1="6492" x2="452" y2="6512" class="solid"></line>
  <polygon points="488,6484 480,6488 488,6492" class="filled"></polygon>
  <line x1="488" y1="6488" x2="500" y2="6488" class="solid end_marked_open_circle"></line>
  <line x1="500" y1="6492" x2="500" y2="6612" class="solid"></line>
  <line x1="504" y1="6488" x2="540" y2="6488" class="solid end_marked_open_circle"></line>
  <line x1="540" y1="6492" x2="540" y2="6528" class="solid"></line>
  <line x1="544" y1="6488" x2="564" y2="6488" class="solid"></line>
  <text x="578" y="6492" >OUT</text>
  <line x1="104" y1="6504" x2="112" y2="6504" class="solid"></line>
  <text x="138" y="6508" >+</text>
  <text x="298" y="6508" >2</text>
  <line x1="292" y1="6520" x2="316" y2="6520" class="solid end_marked_open_circle"></line>
  <text x="450" y="6524" >C</text>
  <line x1="460" y1="6512" x2="460" y2="6560" class="solid"></line>
  <line x1="96" y1="6536" x2="104" y2="6536" class="broken"></line>
  <line x1="112" y1="6536" x2="120" y2="6536" class="broken"></line>
  <line x1="128" y1="6536" x2="136" y2="6536" class="broken"></line>
  <line x1="144" y1="6536" x2="152" y2="6536" class="broken"></line>
  <line x1="160" y1="6536" x2="168" y2="6536" class="broken"></line>
  <text x="306" y="6540" >GND</text>
  <text x="450" y="6540" >C</text>
  <line x1="528" y1="6536" x2="552" y2="6536" class="solid"></line>
  <text x="298" y="6556" >3</text>
  <text x="354" y="6556" >1nF</text>
  <text x="450" y="6556" >C</text>
  <rect x="528" y="6548" width="8" height="8" class="solid filled" rx="0"></rect>
  <rect x="536" y="6548" width="8" height="8" class="solid filled" rx="0"></rect>
  <line x1="540" y1="6556" x2="540" y2="6576" class="solid"></line>
  <rect x="544" y="6548" width="8" height="8" class="solid filled" rx="0"></rect>
  <line x1="452" y1="6560" x2="452" y2="6576" class="solid"></line>
  <text x="554" y="6572" >+</text>
  <text x="442" y="6588" >GND</text>
  <text x="530" y="6588" >GND</text>
  <text x="226" y="6604" >5</text>
  <text x="266" y="6604" >4</text>
  <line x1="280" y1="6616" x2="388" y2="6616" class="solid end_marked_open_circle"></line>
  <line x1="392" y1="6616" x2="500" y2="6616" class="solid end_marked_open_circle"></line>
  <line x1="128" y1="6648" x2="236" y2="6648" class="solid end_marked_circle"></line>
  <line x1="324" y1="6648" x2="428" y2="6648" class="solid end_marked_open_circle"></line>
  <text x="298" y="6668" >2k</text>
  <text x="490" y="6668" >1k0</text>
  <text x="418" y="6764" >GND</text>
  <text x="290" y="6268" >2k2</text>
  <text x="474" y="6300" >LED</text>
  <text x="314" y="6316" >1k</text>
  <text x="402" y="6364" >BC</text>
  <text x="402" y="6380" >547</text>
  <text x="474" y="6428" >IRF9Z34</text>
  <text x="210" y="6460" >MC34063</text>
  <text x="10" y="6508" >6000</text>
  <text x="50" y="6508" >micro</text>
  <text x="10" y="6524" >Farad,</text>
  <text x="66" y="6524" >40V</text>
  <text x="10" y="6540" >Capacitor</text>
  <text x="402" y="6540" >30uH</text>
  <text x="562" y="6540" >470</text>
  <text x="570" y="6556" >uF</text>
  <text x="450" y="6700" >5k6</text>
  <text x="482" y="6700" >+</text>
  <text x="498" y="6700" >3k3</text>
  <text x="450" y="6716" >in</text>
  <text x="474" y="6716" >Serie</text>
  <line x1="16" y1="6824" x2="4" y2="6824" class="solid end_marked_open_circle"></line>
  <polygon points="16,6820 24,6824 16,6828" class="filled"></polygon>
  <text x="34" y="6828" >Latest</text>
  <text x="90" y="6828" >addition:</text>
  <text x="170" y="6828" >Styling</text>
  <text x="234" y="6828" >of</text>
  <text x="258" y="6828" >tagged</text>
  <text x="314" y="6828" >shapes</text>
  <text x="26" y="7388" >Advantages:</text>
  <line x1="8" y1="7416" x2="20" y2="7416" class="solid end_marked_circle"></line>
  <line x1="16" y1="7416" x2="20" y2="7416" class="solid"></line>
  <text x="34" y="7420" >Plain</text>
  <text x="82" y="7420" >text</text>
  <text x="122" y="7420" >format</text>
  <text x="34" y="7436" >Ultimately</text>
  <text x="122" y="7436" >portable,</text>
  <text x="34" y="7452" >Degrades</text>
  <text x="106" y="7452" >gracefully</text>
  <text x="34" y="7468" >Even</text>
  <text x="74" y="7468" >when</text>
  <text x="114" y="7468" >not</text>
  <text x="146" y="7468" >using</text>
  <text x="194" y="7468" >a</text>
  <text x="210" y="7468" >graphical</text>
  <text x="290" y="7468" >renderer,</text>
  <text x="370" y="7468" >it</text>
  <text x="394" y="7468" >would</text>
  <text x="442" y="7468" >still</text>
  <text x="490" y="7468" >looks</text>
  <text x="34" y="7484" >as</text>
  <text x="58" y="7484" >text</text>
  <text x="98" y="7484" >based</text>
  <text x="146" y="7484" >diagrams.</text>
  <text x="226" y="7484" >Paste</text>
  <text x="274" y="7484" >the</text>
  <text x="306" y="7484" >text</text>
  <text x="346" y="7484" >in</text>
  <text x="370" y="7484" >your</text>
  <text x="410" y="7484" >source</text>
  <text x="466" y="7484" >code.</text>
  <text x="34" y="7500" >Easiest</text>
  <text x="98" y="7500" >to</text>
  <text x="122" y="7500" >use.</text>
  <text x="162" y="7500" >Anyone</text>
  <text x="218" y="7500" >knows</text>
  <text x="266" y="7500" >how</text>
  <text x="298" y="7500" >to</text>
  <text x="322" y="7500" >edit</text>
  <text x="362" y="7500" >text.</text>
  <text x="202" y="7436" >backward</text>
  <text x="274" y="7436" >compatible</text>
  <text x="362" y="7436" >and</text>
  <text x="394" y="7436" >future</text>
  <text x="450" y="7436" >proof.</text>
  <line x1="8" y1="7448" x2="20" y2="7448" class="solid end_marked_circle"></line>
  <line x1="16" y1="7448" x2="20" y2="7448" class="solid"></line>
  <text x="538" y="7468" >good</text>
  <line x1="8" y1="7496" x2="20" y2="7496" class="solid end_marked_circle"></line>
  <line x1="16" y1="7496" x2="20" y2="7496" class="solid"></line>
  <text x="362" y="748" >.--------------.</text>
  <text x="362" y="764" >| Don&#39;t draw me|</text>
  <text x="362" y="780" >|              |</text>
  <text x="362" y="796" >&#39;--------------&#39;</text>
  <text x="2" y="3276" >Udc_OK</text>
  <g>
    <path d="M 64,72 A 8,8 0,0,0 58,76" class="nofill"></path>
    <line x1="64" y1="72" x2="88" y2="72" class="solid"></line>
    <path d="M 88,72 A 3,3 0,0,1 90,76" class="nofill"></path>
    <line x1="90" y1="76" x2="64" y2="128" class="solid"></line>
    <line x1="76" y1="104" x2="88" y2="104" class="solid"></line>
    <line x1="66" y1="92" x2="56" y2="112" class="solid"></line>
    <line x1="56" y1="112" x2="72" y2="144" class="solid"></line>
    <path d="M 32,104 A 8,8 0,0,0 26,108" class="nofill"></path>
    <line x1="26" y1="108" x2="24" y2="112" class="solid"></line>
    <line x1="32" y1="104" x2="44" y2="104" class="solid"></line>
    <path d="M 24,112 A 16,16 0,0,0 24,128" class="nofill"></path>
    <line x1="24" y1="128" x2="26" y2="132" class="solid"></line>
    <path d="M 26,132 A 8,8 0,0,0 32,136" class="nofill"></path>
    <line x1="32" y1="136" x2="40" y2="136" class="solid"></line>
    <path d="M 40,136 A 8,8 0,0,1 46,140" class="nofill"></path>
    <line x1="46" y1="140" x2="60" y2="168" class="solid"></line>
    <line x1="72" y1="144" x2="60" y2="168" class="solid"></line>
  </g>
  <g>
    <path d="M 40,280 A 4,4 0,0,0 36,284" class="nofill"></path>
    <line x1="36" y1="284" x2="36" y2="836" class="broken"></line>
    <line x1="40" y1="280" x2="48" y2="280" class="solid"></line>
    <path d="M 36,836 A 4,4 0,0,0 40,840" class="nofill"></path>
    <line x1="40" y1="840" x2="560" y2="840" class="broken"></line>
    <line x1="552" y1="280" x2="560" y2="280" class="solid"></line>
    <path d="M 560,280 A 4,4 0,0,1 564,284" class="nofill"></path>
    <line x1="564" y1="284" x2="564" y2="836" class="broken"></line>
    <path d="M 564,836 A 4,4 0,0,1 560,840" class="nofill"></path>
  </g>
  <g>
    <line x1="460" y1="296" x2="428" y2="328" class="broken"></line>
    <line x1="460" y1="296" x2="492" y2="328" class="broken"></line>
    <line x1="428" y1="328" x2="460" y2="360" class="broken"></line>
    <line x1="492" y1="328" x2="460" y2="360" class="broken"></line>
  </g>
  <g>
    <line x1="80" y1="368" x2="136" y2="368" class="solid"></line>
    <line x1="80" y1="368" x2="64" y2="400" class="solid"></line>
    <line x1="136" y1="368" x2="152" y2="400" class="solid"></line>
    <line x1="64" y1="400" x2="80" y2="432" class="solid"></line>
    <line x1="152" y1="400" x2="136" y2="432" class="solid"></line>
    <line x1="80" y1="432" x2="136" y2="432" class="solid"></line>
  </g>
  <g>
    <path d="M 256,312 A 8,8 0,0,0 250,316" class="nofill"></path>
    <line x1="250" y1="316" x2="248" y2="320" class="solid"></line>
    <line x1="256" y1="312" x2="304" y2="312" class="solid"></line>
    <path d="M 304,312 A 8,8 0,0,1 310,316" class="nofill"></path>
    <line x1="310" y1="316" x2="312" y2="320" class="solid"></line>
    <path d="M 248,320 A 16,16 0,0,0 248,336" class="nofill"></path>
    <path d="M 312,320 A 16,16 0,0,1 312,336" class="nofill"></path>
    <line x1="248" y1="336" x2="250" y2="340" class="solid"></line>
    <path d="M 250,340 A 8,8 0,0,0 256,344" class="nofill"></path>
    <line x1="256" y1="344" x2="304" y2="344" class="solid"></line>
    <line x1="312" y1="336" x2="310" y2="340" class="solid"></line>
    <path d="M 310,340 A 8,8 0,0,1 304,344" class="nofill"></path>
  </g>
  <g>
    <line x1="232" y1="368" x2="296" y2="368" class="solid"></line>
    <line x1="232" y1="368" x2="240" y2="384" class="solid"></line>
    <line x1="296" y1="368" x2="304" y2="384" class="solid"></line>
    <path d="M 240,384 A 16,16 0,0,1 240,400" class="nofill"></path>
    <path d="M 304,384 A 16,16 0,0,1 304,400" class="nofill"></path>
    <line x1="240" y1="400" x2="232" y2="416" class="solid"></line>
    <line x1="232" y1="416" x2="296" y2="416" class="solid"></line>
    <line x1="304" y1="400" x2="296" y2="416" class="solid"></line>
  </g>
  <g>
    <line x1="368" y1="304" x2="350" y2="340" class="solid"></line>
    <line x1="368" y1="304" x2="386" y2="340" class="solid"></line>
    <path d="M 350,340 A 3,3 0,0,0 352,344" class="nofill"></path>
    <line x1="352" y1="344" x2="384" y2="344" class="solid"></line>
    <path d="M 386,340 A 3,3 0,0,1 384,344" class="nofill"></path>
  </g>
  <g>
    <line x1="200" y1="368" x2="184" y2="400" class="solid"></line>
    <line x1="200" y1="368" x2="216" y2="400" class="solid"></line>
    <line x1="184" y1="400" x2="200" y2="432" class="solid"></line>
    <line x1="216" y1="400" x2="200" y2="432" class="solid"></line>
  </g>
  <g>
    <line x1="88" y1="472" x2="168" y2="472" class="solid"></line>
    <path d="M 168,472 A 8,8 0,0,1 174,476" class="nofill"></path>
    <line x1="174" y1="476" x2="176" y2="480" class="solid"></line>
    <path d="M 176,480 A 16,16 0,0,1 176,496" class="nofill"></path>
    <line x1="176" y1="496" x2="174" y2="500" class="solid"></line>
    <path d="M 144,504 A 8,8 0,0,0 138,508" class="nofill"></path>
    <line x1="138" y1="508" x2="128" y2="528" class="solid"></line>
    <line x1="144" y1="504" x2="168" y2="504" class="solid"></line>
    <path d="M 174,500 A 8,8 0,0,1 168,504" class="nofill"></path>
  </g>
  <g>
    <line x1="88" y1="504" x2="124" y2="504" class="solid"></line>
    <line x1="124" y1="504" x2="124" y2="532" class="solid"></line>
  </g>
  <g>
    <line x1="104" y1="560" x2="120" y2="560" class="solid"></line>
    <path d="M 104,560 A 16,16 0,0,0 104,576" class="nofill"></path>
    <line x1="104" y1="576" x2="120" y2="576" class="solid"></line>
    <path d="M 120,560 A 16,16 0,0,1 120,576" class="nofill"></path>
  </g>
  <g>
    <path d="M 240,472 A 8,8 0,0,0 234,476" class="nofill"></path>
    <line x1="234" y1="476" x2="232" y2="480" class="solid"></line>
    <path d="M 232,480 A 16,16 0,0,0 232,496" class="nofill"></path>
    <line x1="232" y1="496" x2="234" y2="500" class="solid"></line>
    <path d="M 234,500 A 8,8 0,0,0 240,504" class="nofill"></path>
  </g>
  <g>
    <path d="M 272,472 A 8,8 0,0,1 278,476" class="nofill"></path>
    <line x1="278" y1="476" x2="280" y2="480" class="solid"></line>
    <path d="M 280,480 A 16,16 0,0,1 280,496" class="nofill"></path>
    <line x1="280" y1="496" x2="278" y2="500" class="solid"></line>
    <path d="M 278,500 A 8,8 0,0,1 272,504" class="nofill"></path>
  </g>
  <g>
    <path d="M 336,472 A 8,8 0,0,0 330,476" class="nofill"></path>
    <line x1="330" y1="476" x2="328" y2="480" class="solid"></line>
    <path d="M 328,480 A 16,16 0,0,0 328,496" class="nofill"></path>
    <line x1="328" y1="496" x2="330" y2="500" class="solid"></line>
    <path d="M 330,500 A 8,8 0,0,0 336,504" class="nofill"></path>
  </g>
  <g>
    <path d="M 360,472 A 8,8 0,0,1 366,476" class="nofill"></path>
    <line x1="366" y1="476" x2="368" y2="480" class="solid"></line>
    <path d="M 368,480 A 16,16 0,0,1 368,496" class="nofill"></path>
    <line x1="368" y1="496" x2="366" y2="500" class="solid"></line>
    <path d="M 366,500 A 8,8 0,0,1 360,504" class="nofill"></path>
  </g>
  <g>
    <line x1="280" y1="552" x2="296" y2="552" class="solid"></line>
    <path d="M 296,552 A 8,8 0,0,1 302,556" class="nofill"></path>
    <line x1="302" y1="556" x2="304" y2="560" class="solid"></line>
    <path d="M 304,560 A 16,16 0,0,1 304,576" class="nofill"></path>
    <line x1="304" y1="576" x2="302" y2="580" class="solid"></line>
    <line x1="280" y1="584" x2="296" y2="584" class="solid"></line>
    <path d="M 302,580 A 8,8 0,0,1 296,584" class="nofill"></path>
  </g>
  <g>
    <path d="M 216,744 A 3,3 0,0,0 214,748" class="nofill"></path>
    <line x1="214" y1="748" x2="236" y2="792" class="solid"></line>
    <line x1="216" y1="744" x2="256" y2="744" class="solid"></line>
    <path d="M 256,744 A 3,3 0,0,1 258,748" class="nofill"></path>
    <line x1="258" y1="748" x2="236" y2="792" class="solid"></line>
  </g>
  <g>
    <path d="M 280,744 A 3,3 0,0,0 278,748" class="nofill"></path>
    <line x1="278" y1="748" x2="298" y2="788" class="solid"></line>
    <line x1="280" y1="744" x2="312" y2="744" class="solid"></line>
    <path d="M 312,744 A 8,8 0,0,1 318,748" class="nofill"></path>
    <line x1="318" y1="748" x2="338" y2="788" class="solid"></line>
    <path d="M 298,788 A 8,8 0,0,0 304,792" class="nofill"></path>
    <line x1="304" y1="792" x2="336" y2="792" class="solid"></line>
    <path d="M 338,788 A 3,3 0,0,1 336,792" class="nofill"></path>
  </g>
  <g>
    <path d="M 72,904 A 8,8 0,0,0 66,908" class="nofill"></path>
    <line x1="72" y1="904" x2="96" y2="904" class="solid"></line>
    <path d="M 96,904 A 3,3 0,0,1 98,908" class="nofill"></path>
    <line x1="98" y1="908" x2="72" y2="960" class="solid"></line>
    <line x1="84" y1="936" x2="96" y2="936" class="solid"></line>
    <line x1="74" y1="924" x2="64" y2="944" class="solid"></line>
    <line x1="64" y1="944" x2="80" y2="976" class="solid"></line>
    <path d="M 40,936 A 8,8 0,0,0 34,940" class="nofill"></path>
    <line x1="34" y1="940" x2="32" y2="944" class="solid"></line>
    <line x1="40" y1="936" x2="52" y2="936" class="solid"></line>
    <path d="M 32,944 A 16,16 0,0,0 32,960" class="nofill"></path>
    <line x1="32" y1="960" x2="34" y2="964" class="solid"></line>
    <path d="M 34,964 A 8,8 0,0,0 40,968" class="nofill"></path>
    <line x1="40" y1="968" x2="48" y2="968" class="solid"></line>
    <path d="M 48,968 A 8,8 0,0,1 54,972" class="nofill"></path>
    <line x1="54" y1="972" x2="68" y2="1000" class="solid"></line>
    <line x1="80" y1="976" x2="68" y2="1000" class="solid"></line>
  </g>
  <g>
    <line x1="276" y1="912" x2="292" y2="912" class="solid"></line>
    <line x1="276" y1="912" x2="276" y2="928" class="solid"></line>
    <line x1="276" y1="920" x2="292" y2="920" class="solid"></line>
    <line x1="292" y1="912" x2="292" y2="968" class="solid"></line>
    <path d="M 276,928 A 12,12 0,0,1 276,944" class="nofill"></path>
    <line x1="276" y1="952" x2="292" y2="952" class="solid"></line>
    <line x1="292" y1="968" x2="280" y2="992" class="solid"></line>
    <path d="M 256,936 A 4,4 0,0,0 252,940" class="nofill"></path>
    <line x1="252" y1="940" x2="252" y2="964" class="solid"></line>
    <line x1="256" y1="936" x2="292" y2="936" class="solid"></line>
    <path d="M 252,964 A 16,16 0,0,0 254,972" class="nofill"></path>
    <line x1="254" y1="972" x2="264" y2="992" class="solid"></line>
    <line x1="264" y1="992" x2="280" y2="992" class="solid"></line>
  </g>
  <g>
    <path d="M 176,920 A 8,8 0,0,0 170,924" class="nofill"></path>
    <line x1="170" y1="924" x2="144" y2="976" class="solid"></line>
    <line x1="176" y1="920" x2="192" y2="920" class="solid"></line>
    <path d="M 192,920 A 8,8 0,0,1 198,924" class="nofill"></path>
    <line x1="198" y1="924" x2="204" y2="936" class="solid"></line>
    <line x1="164" y1="936" x2="184" y2="936" class="solid"></line>
    <path d="M 184,936 A 8,8 0,0,1 190,940" class="nofill"></path>
    <line x1="190" y1="940" x2="192" y2="944" class="solid"></line>
    <line x1="204" y1="936" x2="204" y2="952" class="solid"></line>
    <path d="M 192,944 A 16,16 0,0,1 192,960" class="nofill"></path>
    <line x1="204" y1="952" x2="182" y2="996" class="solid"></line>
    <line x1="152" y1="960" x2="162" y2="980" class="solid"></line>
    <line x1="144" y1="976" x2="154" y2="996" class="solid"></line>
    <path d="M 162,980 A 8,8 0,0,0 168,984" class="nofill"></path>
    <path d="M 154,996 A 8,8 0,0,0 160,1000" class="nofill"></path>
    <line x1="160" y1="1000" x2="176" y2="1000" class="solid"></line>
    <path d="M 182,996 A 8,8 0,0,1 176,1000" class="nofill"></path>
    <line x1="172" y1="960" x2="172" y2="984" class="solid"></line>
    <line x1="172" y1="968" x2="188" y2="968" class="solid"></line>
    <line x1="192" y1="960" x2="182" y2="980" class="solid"></line>
    <line x1="168" y1="984" x2="176" y2="984" class="solid"></line>
    <path d="M 182,980 A 8,8 0,0,1 176,984" class="nofill"></path>
  </g>
  <g>
    <path d="M 424,920 A 8,8 0,0,0 418,924" class="nofill"></path>
    <line x1="418" y1="924" x2="408" y2="944" class="solid"></line>
    <line x1="424" y1="920" x2="440" y2="920" class="solid"></line>
    <path d="M 440,920 A 8,8 0,0,1 446,924" class="nofill"></path>
    <line x1="446" y1="924" x2="456" y2="944" class="solid"></line>
    <path d="M 408,944 A 16,16 0,0,0 408,960" class="nofill"></path>
    <path d="M 456,944 A 16,16 0,0,1 456,960" class="nofill"></path>
    <line x1="408" y1="960" x2="418" y2="980" class="solid"></line>
    <line x1="456" y1="960" x2="446" y2="980" class="solid"></line>
    <path d="M 418,980 A 8,8 0,0,0 424,984" class="nofill"></path>
    <line x1="424" y1="984" x2="440" y2="984" class="solid"></line>
    <path d="M 446,980 A 8,8 0,0,1 440,984" class="nofill"></path>
  </g>
  <g>
    <line x1="424" y1="928" x2="412" y2="952" class="solid"></line>
    <line x1="412" y1="952" x2="424" y2="976" class="solid"></line>
  </g>
  <g>
    <path d="M 432,936 A 8,8 0,0,0 426,940" class="nofill"></path>
    <line x1="426" y1="940" x2="424" y2="944" class="solid"></line>
    <line x1="432" y1="936" x2="440" y2="936" class="solid"></line>
    <path d="M 440,936 A 8,8 0,0,1 446,940" class="nofill"></path>
    <line x1="446" y1="940" x2="448" y2="944" class="solid"></line>
    <path d="M 424,944 A 16,16 0,0,0 424,960" class="nofill"></path>
    <path d="M 448,944 A 16,16 0,0,1 448,960" class="nofill"></path>
    <line x1="424" y1="960" x2="426" y2="964" class="solid"></line>
    <path d="M 426,964 A 8,8 0,0,0 432,968" class="nofill"></path>
    <line x1="432" y1="968" x2="440" y2="968" class="solid"></line>
    <line x1="448" y1="960" x2="446" y2="964" class="solid"></line>
    <path d="M 446,964 A 8,8 0,0,1 440,968" class="nofill"></path>
  </g>
  <g>
    <path d="M 336,936 A 4,4 0,0,0 332,940" class="nofill"></path>
    <line x1="332" y1="940" x2="332" y2="964" class="solid"></line>
    <line x1="336" y1="936" x2="352" y2="936" class="solid"></line>
    <path d="M 352,936 A 4,4 0,0,1 356,940" class="nofill"></path>
    <path d="M 360,936 A 4,4 0,0,0 356,940" class="nofill"></path>
    <line x1="356" y1="940" x2="356" y2="964" class="solid"></line>
    <line x1="360" y1="936" x2="368" y2="936" class="solid"></line>
    <path d="M 368,936 A 4,4 0,0,1 372,940" class="nofill"></path>
    <line x1="372" y1="940" x2="372" y2="964" class="solid"></line>
    <line x1="332" y1="952" x2="344" y2="952" class="solid"></line>
    <path d="M 332,964 A 4,4 0,0,0 336,968" class="nofill"></path>
    <line x1="336" y1="968" x2="352" y2="968" class="solid"></line>
    <path d="M 356,964 A 4,4 0,0,1 352,968" class="nofill"></path>
    <path d="M 356,964 A 4,4 0,0,0 360,968" class="nofill"></path>
    <line x1="360" y1="968" x2="368" y2="968" class="solid"></line>
    <path d="M 372,964 A 4,4 0,0,1 368,968" class="nofill"></path>
  </g>
  <g>
    <line x1="344" y1="1024" x2="384" y2="1024" class="solid"></line>
    <line x1="344" y1="1024" x2="340" y2="1032" class="solid"></line>
    <line x1="340" y1="1032" x2="340" y2="1072" class="solid"></line>
    <line x1="312" y1="1072" x2="340" y2="1072" class="solid"></line>
    <line x1="312" y1="1072" x2="296" y2="1104" class="solid"></line>
    <line x1="296" y1="1104" x2="312" y2="1136" class="solid"></line>
    <line x1="368" y1="1040" x2="384" y2="1040" class="solid"></line>
    <path d="M 384,1024 A 16,16 0,0,1 384,1040" class="nofill"></path>
    <line x1="368" y1="1040" x2="364" y2="1048" class="solid"></line>
    <line x1="364" y1="1048" x2="364" y2="1072" class="solid"></line>
    <line x1="384" y1="1040" x2="392" y2="1056" class="solid"></line>
    <line x1="364" y1="1072" x2="384" y2="1072" class="solid"></line>
    <line x1="392" y1="1056" x2="384" y2="1072" class="solid"></line>
  </g>
  <g>
    <line x1="392" y1="1024" x2="408" y2="1056" class="solid"></line>
    <line x1="408" y1="1056" x2="392" y2="1088" class="solid"></line>
    <line x1="384" y1="1088" x2="392" y2="1088" class="solid"></line>
  </g>
  <g>
    <path d="M 328,1072 A 16,16 0,0,0 328,1088" class="nofill"></path>
    <line x1="328" y1="1088" x2="340" y2="1088" class="solid"></line>
    <line x1="340" y1="1088" x2="340" y2="1104" class="solid"></line>
  </g>
  <g>
    <line x1="364" y1="1088" x2="376" y2="1088" class="solid"></line>
    <path d="M 376,1072 A 16,16 0,0,1 376,1088" class="nofill"></path>
    <line x1="364" y1="1088" x2="364" y2="1120" class="solid"></line>
  </g>
  <g>
    <line x1="320" y1="1088" x2="312" y2="1104" class="solid"></line>
    <line x1="312" y1="1104" x2="320" y2="1120" class="solid"></line>
    <line x1="320" y1="1120" x2="336" y2="1120" class="solid"></line>
    <line x1="344" y1="1104" x2="336" y2="1120" class="solid"></line>
    <path d="M 320,1120 A 16,16 0,0,0 320,1136" class="nofill"></path>
    <line x1="320" y1="1136" x2="360" y2="1136" class="solid"></line>
    <line x1="368" y1="1120" x2="360" y2="1136" class="solid"></line>
  </g>
  <g>
    <path d="M 64,1032 A 4,4 0,0,0 60,1036" class="nofill"></path>
    <line x1="60" y1="1036" x2="60" y2="1072" class="solid"></line>
    <line x1="64" y1="1032" x2="96" y2="1032" class="solid"></line>
    <path d="M 96,1032 A 4,4 0,0,1 100,1036" class="nofill"></path>
    <line x1="100" y1="1036" x2="100" y2="1072" class="solid"></line>
    <line x1="36" y1="1072" x2="124" y2="1072" class="solid"></line>
    <line x1="36" y1="1072" x2="36" y2="1092" class="solid"></line>
    <path d="M 36,1092 A 4,4 0,0,0 40,1096" class="nofill"></path>
    <line x1="40" y1="1096" x2="52" y2="1096" class="solid"></line>
    <line x1="56" y1="1088" x2="104" y2="1088" class="solid"></line>
    <line x1="56" y1="1088" x2="46" y2="1108" class="solid"></line>
    <line x1="104" y1="1088" x2="114" y2="1108" class="solid"></line>
    <line x1="108" y1="1096" x2="120" y2="1096" class="solid"></line>
    <path d="M 46,1108 A 3,3 0,0,0 48,1112" class="nofill"></path>
    <line x1="48" y1="1112" x2="112" y2="1112" class="solid"></line>
    <path d="M 114,1108 A 3,3 0,0,1 112,1112" class="nofill"></path>
    <line x1="104" y1="1080" x2="124" y2="1080" class="solid"></line>
    <line x1="124" y1="1072" x2="124" y2="1092" class="solid"></line>
    <path d="M 124,1092 A 4,4 0,0,1 120,1096" class="nofill"></path>
  </g>
  <g>
    <line x1="224" y1="1040" x2="232" y2="1040" class="solid"></line>
    <path d="M 224,1040 A 16,16 0,0,0 224,1056" class="nofill"></path>
    <line x1="224" y1="1056" x2="232" y2="1056" class="solid"></line>
    <path d="M 232,1040 A 16,16 0,0,1 232,1056" class="nofill"></path>
    <line x1="232" y1="1056" x2="240" y2="1072" class="solid"></line>
    <path d="M 240,1072 A 16,16 0,0,1 240,1088" class="nofill"></path>
    <line x1="224" y1="1104" x2="232" y2="1104" class="solid"></line>
    <line x1="240" y1="1088" x2="232" y2="1104" class="solid"></line>
    <path d="M 224,1104 A 16,16 0,0,0 224,1120" class="nofill"></path>
    <line x1="224" y1="1120" x2="232" y2="1120" class="solid"></line>
    <path d="M 232,1104 A 16,16 0,0,1 232,1120" class="nofill"></path>
  </g>
  <g>
    <path d="M 200,1048 A 8,8 0,0,0 194,1052" class="nofill"></path>
    <line x1="194" y1="1052" x2="184" y2="1072" class="solid"></line>
    <line x1="200" y1="1048" x2="216" y2="1048" class="solid"></line>
    <line x1="176" y1="1072" x2="184" y2="1072" class="solid"></line>
    <path d="M 176,1072 A 16,16 0,0,0 176,1088" class="nofill"></path>
    <line x1="176" y1="1088" x2="184" y2="1088" class="solid"></line>
    <path d="M 184,1072 A 16,16 0,0,1 184,1088" class="nofill"></path>
    <line x1="184" y1="1088" x2="194" y2="1108" class="solid"></line>
    <path d="M 194,1108 A 8,8 0,0,0 200,1112" class="nofill"></path>
    <line x1="200" y1="1112" x2="216" y2="1112" class="solid"></line>
  </g>
  <g>
    <path d="M 208,1064 A 8,8 0,0,0 202,1068" class="nofill"></path>
    <line x1="202" y1="1068" x2="200" y2="1072" class="solid"></line>
    <line x1="208" y1="1064" x2="216" y2="1064" class="solid"></line>
    <path d="M 216,1064 A 8,8 0,0,1 222,1068" class="nofill"></path>
    <line x1="222" y1="1068" x2="224" y2="1072" class="solid"></line>
    <path d="M 200,1072 A 16,16 0,0,0 200,1088" class="nofill"></path>
    <path d="M 224,1072 A 16,16 0,0,1 224,1088" class="nofill"></path>
    <line x1="200" y1="1088" x2="202" y2="1092" class="solid"></line>
    <path d="M 202,1092 A 8,8 0,0,0 208,1096" class="nofill"></path>
    <line x1="208" y1="1096" x2="216" y2="1096" class="solid"></line>
    <line x1="224" y1="1088" x2="222" y2="1092" class="solid"></line>
    <path d="M 222,1092 A 8,8 0,0,1 216,1096" class="nofill"></path>
  </g>
  <g>
    <line x1="100" y1="1176" x2="124" y2="1176" class="solid"></line>
    <line x1="100" y1="1176" x2="100" y2="1208" class="solid"></line>
    <line x1="116" y1="1176" x2="116" y2="1208" class="solid"></line>
    <line x1="124" y1="1176" x2="124" y2="1208" class="solid"></line>
    <line x1="100" y1="1192" x2="124" y2="1192" class="solid"></line>
    <line x1="100" y1="1208" x2="124" y2="1208" class="solid"></line>
  </g>
  <g>
    <path d="M 104,1224 A 4,4 0,0,0 100,1228" class="nofill"></path>
    <line x1="100" y1="1228" x2="100" y2="1252" class="solid"></line>
    <line x1="104" y1="1224" x2="120" y2="1224" class="solid"></line>
    <line x1="116" y1="1224" x2="116" y2="1256" class="solid"></line>
    <path d="M 120,1224 A 4,4 0,0,1 124,1228" class="nofill"></path>
    <line x1="124" y1="1228" x2="124" y2="1252" class="solid"></line>
    <line x1="100" y1="1240" x2="124" y2="1240" class="solid"></line>
    <path d="M 100,1252 A 4,4 0,0,0 104,1256" class="nofill"></path>
    <line x1="104" y1="1256" x2="120" y2="1256" class="solid"></line>
    <path d="M 124,1252 A 4,4 0,0,1 120,1256" class="nofill"></path>
  </g>
  <g>
    <line x1="194" y1="1176" x2="222" y2="1176" class="solid"></line>
    <line x1="194" y1="1176" x2="194" y2="1208" class="solid"></line>
    <line x1="198" y1="1176" x2="198" y2="1208" class="solid"></line>
    <line x1="210" y1="1176" x2="210" y2="1208" class="solid"></line>
    <line x1="214" y1="1176" x2="214" y2="1208" class="solid"></line>
    <line x1="218" y1="1176" x2="218" y2="1208" class="solid"></line>
    <line x1="222" y1="1176" x2="222" y2="1208" class="solid"></line>
    <line x1="198" y1="1192" x2="218" y2="1192" class="solid"></line>
    <line x1="194" y1="1208" x2="222" y2="1208" class="solid"></line>
  </g>
  <g>
    <line x1="244" y1="1176" x2="268" y2="1176" class="solid"></line>
    <line x1="244" y1="1176" x2="244" y2="1212" class="solid"></line>
    <line x1="244" y1="1180" x2="268" y2="1180" class="solid"></line>
    <line x1="260" y1="1180" x2="260" y2="1208" class="solid"></line>
    <line x1="268" y1="1176" x2="268" y2="1212" class="solid"></line>
    <line x1="244" y1="1192" x2="268" y2="1192" class="solid"></line>
    <line x1="244" y1="1196" x2="268" y2="1196" class="solid"></line>
    <line x1="244" y1="1208" x2="268" y2="1208" class="solid"></line>
    <line x1="244" y1="1212" x2="268" y2="1212" class="solid"></line>
  </g>
  <g>
    <path d="M 40,1624 A 8,8 0,0,0 34,1628" class="nofill"></path>
    <line x1="34" y1="1628" x2="20" y2="1656" class="solid"></line>
    <line x1="40" y1="1624" x2="72" y2="1624" class="solid"></line>
    <path d="M 72,1624 A 8,8 0,0,1 78,1628" class="nofill"></path>
    <line x1="78" y1="1628" x2="92" y2="1656" class="solid"></line>
    <line x1="20" y1="1656" x2="36" y2="1688" class="solid"></line>
    <path d="M 152,1624 A 8,8 0,0,0 146,1628" class="nofill"></path>
    <line x1="146" y1="1628" x2="132" y2="1656" class="solid"></line>
    <line x1="152" y1="1624" x2="184" y2="1624" class="solid"></line>
    <path d="M 184,1624 A 8,8 0,0,1 190,1628" class="nofill"></path>
    <line x1="190" y1="1628" x2="204" y2="1656" class="solid"></line>
    <line x1="92" y1="1656" x2="132" y2="1656" class="solid"></line>
    <line x1="92" y1="1656" x2="76" y2="1688" class="solid"></line>
    <line x1="132" y1="1656" x2="148" y2="1688" class="solid"></line>
    <line x1="204" y1="1656" x2="240" y2="1656" class="solid"></line>
    <line x1="204" y1="1656" x2="188" y2="1688" class="solid"></line>
    <path d="M 240,1656 A 8,8 0,0,1 246,1660" class="nofill"></path>
    <line x1="246" y1="1660" x2="260" y2="1688" class="solid"></line>
    <line x1="36" y1="1688" x2="76" y2="1688" class="solid"></line>
    <line x1="36" y1="1688" x2="20" y2="1720" class="solid"></line>
    <line x1="76" y1="1688" x2="92" y2="1720" class="solid"></line>
    <line x1="148" y1="1688" x2="188" y2="1688" class="solid"></line>
    <line x1="148" y1="1688" x2="132" y2="1720" class="solid"></line>
    <line x1="188" y1="1688" x2="204" y2="1720" class="solid"></line>
    <line x1="260" y1="1688" x2="244" y2="1720" class="solid"></line>
    <line x1="20" y1="1720" x2="34" y2="1748" class="solid"></line>
    <line x1="92" y1="1720" x2="132" y2="1720" class="solid"></line>
    <line x1="92" y1="1720" x2="76" y2="1752" class="solid"></line>
    <line x1="132" y1="1720" x2="148" y2="1752" class="solid"></line>
    <line x1="204" y1="1720" x2="244" y2="1720" class="solid"></line>
    <line x1="204" y1="1720" x2="188" y2="1752" class="solid"></line>
    <line x1="244" y1="1720" x2="260" y2="1752" class="solid"></line>
    <path d="M 34,1748 A 8,8 0,0,0 40,1752" class="nofill"></path>
    <line x1="40" y1="1752" x2="76" y2="1752" class="solid"></line>
    <line x1="76" y1="1752" x2="90" y2="1780" class="solid"></line>
    <line x1="148" y1="1752" x2="188" y2="1752" class="solid"></line>
    <line x1="148" y1="1752" x2="134" y2="1780" class="solid"></line>
    <line x1="188" y1="1752" x2="202" y2="1780" class="solid"></line>
    <line x1="260" y1="1752" x2="246" y2="1780" class="solid"></line>
    <path d="M 90,1780 A 8,8 0,0,0 96,1784" class="nofill"></path>
    <line x1="96" y1="1784" x2="128" y2="1784" class="solid"></line>
    <path d="M 134,1780 A 8,8 0,0,1 128,1784" class="nofill"></path>
    <path d="M 202,1780 A 8,8 0,0,0 208,1784" class="nofill"></path>
    <line x1="208" y1="1784" x2="240" y2="1784" class="solid"></line>
    <path d="M 246,1780 A 8,8 0,0,1 240,1784" class="nofill"></path>
  </g>
  <g>
    <path d="M 304,1640 A 4,4 0,0,0 300,1644" class="nofill"></path>
    <line x1="300" y1="1644" x2="300" y2="1780" class="solid"></line>
    <line x1="304" y1="1640" x2="440" y2="1640" class="solid"></line>
    <line x1="348" y1="1640" x2="348" y2="1784" class="solid"></line>
    <line x1="396" y1="1640" x2="396" y2="1784" class="solid"></line>
    <path d="M 440,1640 A 4,4 0,0,1 444,1644" class="nofill"></path>
    <line x1="444" y1="1644" x2="444" y2="1780" class="solid"></line>
    <line x1="300" y1="1688" x2="444" y2="1688" class="solid"></line>
    <line x1="300" y1="1736" x2="444" y2="1736" class="solid"></line>
    <path d="M 300,1780 A 4,4 0,0,0 304,1784" class="nofill"></path>
    <line x1="304" y1="1784" x2="440" y2="1784" class="solid"></line>
    <path d="M 444,1780 A 4,4 0,0,1 440,1784" class="nofill"></path>
  </g>
  <g>
    <path d="M 536,1656 A 8,8 0,0,0 530,1660" class="nofill"></path>
    <line x1="530" y1="1660" x2="470" y2="1780" class="solid"></line>
    <line x1="536" y1="1656" x2="724" y2="1656" class="solid"></line>
    <line x1="580" y1="1656" x2="516" y2="1784" class="solid"></line>
    <line x1="628" y1="1656" x2="564" y2="1784" class="solid"></line>
    <line x1="676" y1="1656" x2="612" y2="1784" class="solid"></line>
    <line x1="724" y1="1656" x2="660" y2="1784" class="solid"></line>
    <line x1="516" y1="1688" x2="708" y2="1688" class="solid"></line>
    <line x1="500" y1="1720" x2="692" y2="1720" class="solid"></line>
    <line x1="484" y1="1752" x2="676" y2="1752" class="solid"></line>
    <path d="M 470,1780 A 3,3 0,0,0 472,1784" class="nofill"></path>
    <line x1="472" y1="1784" x2="660" y2="1784" class="solid"></line>
  </g>
  <g>
    <line x1="56" y1="1888" x2="80" y2="1888" class="solid"></line>
    <line x1="80" y1="1888" x2="88" y2="1904" class="solid"></line>
    <line x1="88" y1="1904" x2="112" y2="1904" class="solid"></line>
    <line x1="120" y1="1888" x2="144" y2="1888" class="solid"></line>
    <line x1="120" y1="1888" x2="112" y2="1904" class="solid"></line>
    <line x1="144" y1="1888" x2="152" y2="1904" class="solid"></line>
    <line x1="112" y1="1904" x2="120" y2="1920" class="solid"></line>
    <line x1="120" y1="1920" x2="144" y2="1920" class="solid"></line>
    <line x1="152" y1="1904" x2="144" y2="1920" class="solid"></line>
    <line x1="144" y1="1920" x2="152" y2="1936" class="solid"></line>
    <line x1="24" y1="1904" x2="48" y2="1904" class="solid"></line>
    <line x1="56" y1="1888" x2="48" y2="1904" class="solid"></line>
    <line x1="24" y1="1904" x2="16" y2="1920" class="solid"></line>
    <line x1="48" y1="1904" x2="56" y2="1920" class="solid"></line>
    <line x1="56" y1="1920" x2="80" y2="1920" class="solid"></line>
    <line x1="88" y1="1904" x2="80" y2="1920" class="solid"></line>
    <line x1="16" y1="1920" x2="24" y2="1936" class="solid"></line>
    <line x1="24" y1="1936" x2="48" y2="1936" class="solid"></line>
    <line x1="56" y1="1920" x2="48" y2="1936" class="solid"></line>
    <line x1="80" y1="1920" x2="88" y2="1936" class="solid"></line>
    <line x1="88" y1="1936" x2="112" y2="1936" class="solid"></line>
    <line x1="120" y1="1920" x2="112" y2="1936" class="solid"></line>
    <line x1="24" y1="1936" x2="16" y2="1952" class="solid"></line>
    <line x1="48" y1="1936" x2="56" y2="1952" class="solid"></line>
    <line x1="56" y1="1952" x2="80" y2="1952" class="solid"></line>
    <line x1="88" y1="1936" x2="80" y2="1952" class="solid"></line>
    <line x1="112" y1="1936" x2="120" y2="1952" class="solid"></line>
    <line x1="120" y1="1952" x2="144" y2="1952" class="solid"></line>
    <line x1="152" y1="1936" x2="144" y2="1952" class="solid"></line>
    <line x1="16" y1="1952" x2="24" y2="1968" class="solid"></line>
    <line x1="24" y1="1968" x2="48" y2="1968" class="solid"></line>
    <line x1="56" y1="1952" x2="48" y2="1968" class="solid"></line>
    <line x1="80" y1="1952" x2="88" y2="1968" class="solid"></line>
    <line x1="88" y1="1968" x2="112" y2="1968" class="solid"></line>
    <line x1="120" y1="1952" x2="112" y2="1968" class="solid"></line>
    <line x1="144" y1="1952" x2="152" y2="1968" class="solid"></line>
    <line x1="48" y1="1968" x2="56" y2="1984" class="solid"></line>
    <line x1="56" y1="1984" x2="80" y2="1984" class="solid"></line>
    <line x1="88" y1="1968" x2="80" y2="1984" class="solid"></line>
    <line x1="112" y1="1968" x2="120" y2="1984" class="solid"></line>
    <line x1="120" y1="1984" x2="144" y2="1984" class="solid"></line>
    <line x1="152" y1="1968" x2="144" y2="1984" class="solid"></line>
  </g>
  <g>
    <path d="M 200,1880 A 4,4 0,0,0 196,1884" class="nofill"></path>
    <line x1="196" y1="1884" x2="196" y2="1972" class="solid"></line>
    <line x1="200" y1="1880" x2="352" y2="1880" class="solid"></line>
    <line x1="228" y1="1880" x2="228" y2="1976" class="solid"></line>
    <line x1="260" y1="1880" x2="260" y2="1976" class="solid"></line>
    <line x1="292" y1="1880" x2="292" y2="1976" class="solid"></line>
    <line x1="324" y1="1880" x2="324" y2="1976" class="solid"></line>
    <path d="M 352,1880 A 4,4 0,0,1 356,1884" class="nofill"></path>
    <line x1="356" y1="1884" x2="356" y2="1972" class="solid"></line>
    <line x1="196" y1="1912" x2="356" y2="1912" class="solid"></line>
    <line x1="196" y1="1944" x2="356" y2="1944" class="solid"></line>
    <path d="M 196,1972 A 4,4 0,0,0 200,1976" class="nofill"></path>
    <line x1="200" y1="1976" x2="352" y2="1976" class="solid"></line>
    <path d="M 356,1972 A 4,4 0,0,1 352,1976" class="nofill"></path>
  </g>
  <g>
    <line x1="404" y1="1880" x2="388" y2="1912" class="solid"></line>
    <line x1="404" y1="1880" x2="452" y2="1976" class="solid"></line>
    <path d="M 408,1880 A 8,8 0,0,0 402,1884" class="nofill"></path>
    <path d="M 408,1880 A 3,3 0,0,0 406,1884" class="nofill"></path>
    <line x1="408" y1="1880" x2="528" y2="1880" class="solid"></line>
    <line x1="436" y1="1880" x2="390" y2="1972" class="solid"></line>
    <line x1="436" y1="1880" x2="484" y2="1976" class="solid"></line>
    <line x1="468" y1="1880" x2="420" y2="1976" class="solid"></line>
    <line x1="468" y1="1880" x2="516" y2="1976" class="solid"></line>
    <line x1="500" y1="1880" x2="452" y2="1976" class="solid"></line>
    <line x1="500" y1="1880" x2="532" y2="1944" class="solid"></line>
    <path d="M 528,1880 A 3,3 0,0,1 530,1884" class="nofill"></path>
    <line x1="530" y1="1884" x2="484" y2="1976" class="solid"></line>
    <line x1="388" y1="1912" x2="516" y2="1912" class="solid"></line>
    <line x1="388" y1="1912" x2="420" y2="1976" class="solid"></line>
    <line x1="404" y1="1944" x2="532" y2="1944" class="solid"></line>
    <line x1="532" y1="1944" x2="516" y2="1976" class="solid"></line>
    <path d="M 390,1972 A 3,3 0,0,0 392,1976" class="nofill"></path>
    <line x1="392" y1="1976" x2="512" y2="1976" class="solid"></line>
    <path d="M 514,1972 A 3,3 0,0,1 512,1976" class="nofill"></path>
    <path d="M 518,1972 A 8,8 0,0,1 512,1976" class="nofill"></path>
  </g>
  <g>
    <path d="M 560,1880 A 4,4 0,0,0 556,1884" class="nofill"></path>
    <line x1="556" y1="1884" x2="556" y2="1972" class="solid"></line>
    <line x1="560" y1="1880" x2="584" y2="1880" class="solid"></line>
    <path d="M 584,1880 A 4,4 0,0,1 588,1884" class="nofill"></path>
    <line x1="588" y1="1884" x2="588" y2="1972" class="solid"></line>
    <line x1="556" y1="1912" x2="588" y2="1912" class="solid"></line>
    <line x1="556" y1="1944" x2="588" y2="1944" class="solid"></line>
    <path d="M 556,1972 A 4,4 0,0,0 560,1976" class="nofill"></path>
    <line x1="560" y1="1976" x2="584" y2="1976" class="solid"></line>
    <path d="M 588,1972 A 4,4 0,0,1 584,1976" class="nofill"></path>
    <path d="M 624,1880 A 4,4 0,0,0 620,1884" class="nofill"></path>
    <line x1="620" y1="1884" x2="620" y2="1972" class="solid"></line>
    <line x1="624" y1="1880" x2="648" y2="1880" class="solid"></line>
    <path d="M 648,1880 A 4,4 0,0,1 652,1884" class="nofill"></path>
    <line x1="652" y1="1884" x2="652" y2="1972" class="solid"></line>
    <line x1="588" y1="1896" x2="620" y2="1896" class="solid"></line>
    <line x1="620" y1="1912" x2="652" y2="1912" class="solid"></line>
    <line x1="588" y1="1928" x2="620" y2="1928" class="solid"></line>
    <line x1="620" y1="1944" x2="652" y2="1944" class="solid"></line>
    <line x1="588" y1="1960" x2="620" y2="1960" class="solid"></line>
    <path d="M 620,1972 A 4,4 0,0,0 624,1976" class="nofill"></path>
    <line x1="624" y1="1976" x2="648" y2="1976" class="solid"></line>
    <path d="M 652,1972 A 4,4 0,0,1 648,1976" class="nofill"></path>
  </g>
  <g>
    <line x1="488" y1="2064" x2="476" y2="2088" class="solid"></line>
    <line x1="476" y1="2088" x2="488" y2="2112" class="solid"></line>
  </g>
  <g>
    <line x1="172" y1="2108" x2="172" y2="2152" class="solid"></line>
    <line x1="172" y1="2152" x2="216" y2="2152" class="solid"></line>
    <line x1="172" y1="2152" x2="158" y2="2180" class="solid"></line>
  </g>
  <g>
    <path d="M 192,2136 A 4,4 0,0,1 196,2140" class="nofill"></path>
    <line x1="196" y1="2140" x2="196" y2="2144" class="solid"></line>
  </g>
  <g>
    <text x="114" y="2316">文</text>
    <text x="130" y="2316">件</text>
    <text x="146" y="2316">系</text>
    <text x="162" y="2316">统</text>
  </g>
  <g>
    <text x="258" y="2316">调</text>
    <text x="274" y="2316">度</text>
    <text x="290" y="2316">器</text>
  </g>
  <g>
    <path d="M 288,2392 A 8,8 0,0,0 282,2396" class="nofill"></path>
    <line x1="282" y1="2396" x2="270" y2="2420" class="solid"></line>
    <line x1="288" y1="2392" x2="312" y2="2392" class="solid"></line>
  </g>
  <g>
    <line x1="270" y1="2428" x2="282" y2="2452" class="solid"></line>
    <path d="M 282,2452 A 8,8 0,0,0 288,2456" class="nofill"></path>
    <line x1="288" y1="2456" x2="312" y2="2456" class="solid"></line>
  </g>
  <g>
    <line x1="258" y1="2444" x2="246" y2="2468" class="solid"></line>
    <line x1="232" y1="2472" x2="240" y2="2472" class="solid"></line>
    <path d="M 246,2468 A 8,8 0,0,1 240,2472" class="nofill"></path>
  </g>
  <g>
    <line x1="220" y1="2544" x2="220" y2="2568" class="solid"></line>
    <line x1="220" y1="2568" x2="208" y2="2592" class="solid"></line>
    <line x1="220" y1="2568" x2="232" y2="2592" class="solid"></line>
  </g>
  <g>
    <path d="M 80,2568 A 4,4 0,0,0 76,2572" class="nofill"></path>
    <line x1="76" y1="2572" x2="76" y2="2596" class="solid"></line>
    <line x1="80" y1="2568" x2="104" y2="2568" class="solid"></line>
    <path d="M 104,2568 A 4,4 0,0,1 108,2572" class="nofill"></path>
    <line x1="108" y1="2572" x2="108" y2="2596" class="solid"></line>
    <path d="M 76,2596 A 4,4 0,0,0 80,2600" class="nofill"></path>
    <line x1="80" y1="2600" x2="104" y2="2600" class="solid"></line>
    <line x1="92" y1="2600" x2="92" y2="2696" class="solid"></line>
    <path d="M 108,2596 A 4,4 0,0,1 104,2600" class="nofill"></path>
    <line x1="92" y1="2632" x2="208" y2="2632" class="solid"></line>
    <path d="M 80,2696 A 4,4 0,0,0 76,2700" class="nofill"></path>
    <line x1="76" y1="2700" x2="76" y2="2724" class="solid"></line>
    <line x1="80" y1="2696" x2="104" y2="2696" class="solid"></line>
    <path d="M 104,2696 A 4,4 0,0,1 108,2700" class="nofill"></path>
    <line x1="108" y1="2700" x2="108" y2="2724" class="solid"></line>
    <path d="M 76,2724 A 4,4 0,0,0 80,2728" class="nofill"></path>
    <line x1="80" y1="2728" x2="104" y2="2728" class="solid"></line>
    <path d="M 108,2724 A 4,4 0,0,1 104,2728" class="nofill"></path>
  </g>
  <g>
    <line x1="220" y1="2736" x2="220" y2="2760" class="solid"></line>
    <line x1="220" y1="2760" x2="208" y2="2784" class="solid"></line>
    <line x1="220" y1="2760" x2="232" y2="2784" class="solid"></line>
  </g>
  <g>
    <path d="M 120,2824 A 8,8 0,0,0 114,2828" class="nofill"></path>
    <line x1="114" y1="2828" x2="112" y2="2832" class="solid"></line>
    <line x1="120" y1="2824" x2="128" y2="2824" class="solid"></line>
    <path d="M 128,2824 A 8,8 0,0,1 134,2828" class="nofill"></path>
    <line x1="134" y1="2828" x2="136" y2="2832" class="solid"></line>
    <path d="M 112,2832 A 16,16 0,0,0 112,2848" class="nofill"></path>
    <path d="M 136,2832 A 16,16 0,0,1 136,2848" class="nofill"></path>
    <line x1="112" y1="2848" x2="114" y2="2852" class="solid"></line>
    <path d="M 114,2852 A 8,8 0,0,0 120,2856" class="nofill"></path>
    <line x1="120" y1="2856" x2="128" y2="2856" class="solid"></line>
    <line x1="136" y1="2848" x2="134" y2="2852" class="solid"></line>
    <path d="M 134,2852 A 8,8 0,0,1 128,2856" class="nofill"></path>
  </g>
  <g>
    <path d="M 80,2920 A 8,8 0,0,0 74,2924" class="nofill"></path>
    <line x1="74" y1="2924" x2="72" y2="2928" class="solid"></line>
    <line x1="80" y1="2920" x2="88" y2="2920" class="solid"></line>
    <path d="M 88,2920 A 8,8 0,0,1 94,2924" class="nofill"></path>
    <line x1="94" y1="2924" x2="96" y2="2928" class="solid"></line>
    <path d="M 72,2928 A 16,16 0,0,0 72,2944" class="nofill"></path>
    <path d="M 96,2928 A 16,16 0,0,1 96,2944" class="nofill"></path>
    <line x1="72" y1="2944" x2="74" y2="2948" class="solid"></line>
    <path d="M 74,2948 A 8,8 0,0,0 80,2952" class="nofill"></path>
    <line x1="80" y1="2952" x2="88" y2="2952" class="solid"></line>
    <line x1="96" y1="2944" x2="94" y2="2948" class="solid"></line>
    <path d="M 94,2948 A 8,8 0,0,1 88,2952" class="nofill"></path>
  </g>
  <g>
    <path d="M 112,3016 A 8,8 0,0,0 106,3020" class="nofill"></path>
    <line x1="106" y1="3020" x2="104" y2="3024" class="solid"></line>
    <line x1="112" y1="3016" x2="120" y2="3016" class="solid"></line>
    <path d="M 120,3016 A 8,8 0,0,1 126,3020" class="nofill"></path>
    <line x1="126" y1="3020" x2="128" y2="3024" class="solid"></line>
    <path d="M 104,3024 A 16,16 0,0,0 104,3040" class="nofill"></path>
    <path d="M 128,3024 A 16,16 0,0,1 128,3040" class="nofill"></path>
    <line x1="104" y1="3040" x2="106" y2="3044" class="solid"></line>
    <path d="M 106,3044 A 8,8 0,0,0 112,3048" class="nofill"></path>
    <line x1="112" y1="3048" x2="120" y2="3048" class="solid"></line>
    <line x1="128" y1="3040" x2="126" y2="3044" class="solid"></line>
    <path d="M 126,3044 A 8,8 0,0,1 120,3048" class="nofill"></path>
  </g>
  <g>
    <path d="M 176,2920 A 8,8 0,0,0 170,2924" class="nofill"></path>
    <line x1="170" y1="2924" x2="168" y2="2928" class="solid"></line>
    <line x1="176" y1="2920" x2="184" y2="2920" class="solid"></line>
    <path d="M 184,2920 A 8,8 0,0,1 190,2924" class="nofill"></path>
    <line x1="190" y1="2924" x2="192" y2="2928" class="solid"></line>
    <path d="M 168,2928 A 16,16 0,0,0 168,2944" class="nofill"></path>
    <path d="M 192,2928 A 16,16 0,0,1 192,2944" class="nofill"></path>
    <line x1="168" y1="2944" x2="170" y2="2948" class="solid"></line>
    <path d="M 170,2948 A 8,8 0,0,0 176,2952" class="nofill"></path>
    <line x1="176" y1="2952" x2="184" y2="2952" class="solid"></line>
    <line x1="192" y1="2944" x2="190" y2="2948" class="solid"></line>
    <path d="M 190,2948 A 8,8 0,0,1 184,2952" class="nofill"></path>
  </g>
  <g>
    <path d="M 176,3032 A 8,8 0,0,0 170,3036" class="nofill"></path>
    <line x1="170" y1="3036" x2="168" y2="3040" class="solid"></line>
    <line x1="176" y1="3032" x2="184" y2="3032" class="solid"></line>
    <path d="M 184,3032 A 8,8 0,0,1 190,3036" class="nofill"></path>
    <line x1="190" y1="3036" x2="192" y2="3040" class="solid"></line>
    <path d="M 168,3040 A 16,16 0,0,0 168,3056" class="nofill"></path>
    <path d="M 192,3040 A 16,16 0,0,1 192,3056" class="nofill"></path>
    <line x1="168" y1="3056" x2="170" y2="3060" class="solid"></line>
    <path d="M 170,3060 A 8,8 0,0,0 176,3064" class="nofill"></path>
    <line x1="176" y1="3064" x2="184" y2="3064" class="solid"></line>
    <line x1="192" y1="3056" x2="190" y2="3060" class="solid"></line>
    <path d="M 190,3060 A 8,8 0,0,1 184,3064" class="nofill"></path>
  </g>
  <g>
    <path d="M 224,3032 A 8,8 0,0,0 218,3036" class="nofill"></path>
    <line x1="218" y1="3036" x2="216" y2="3040" class="solid"></line>
    <line x1="224" y1="3032" x2="232" y2="3032" class="solid"></line>
    <path d="M 232,3032 A 8,8 0,0,1 238,3036" class="nofill"></path>
    <line x1="238" y1="3036" x2="240" y2="3040" class="solid"></line>
    <path d="M 216,3040 A 16,16 0,0,0 216,3056" class="nofill"></path>
    <path d="M 240,3040 A 16,16 0,0,1 240,3056" class="nofill"></path>
    <line x1="216" y1="3056" x2="218" y2="3060" class="solid"></line>
    <path d="M 218,3060 A 8,8 0,0,0 224,3064" class="nofill"></path>
    <line x1="224" y1="3064" x2="232" y2="3064" class="solid"></line>
    <line x1="240" y1="3056" x2="238" y2="3060" class="solid"></line>
    <path d="M 238,3060 A 8,8 0,0,1 232,3064" class="nofill"></path>
  </g>
  <g>
    <path d="M 272,3032 A 8,8 0,0,0 266,3036" class="nofill"></path>
    <line x1="266" y1="3036" x2="264" y2="3040" class="solid"></line>
    <line x1="272" y1="3032" x2="280" y2="3032" class="solid"></line>
    <path d="M 280,3032 A 8,8 0,0,1 286,3036" class="nofill"></path>
    <line x1="286" y1="3036" x2="288" y2="3040" class="solid"></line>
    <path d="M 264,3040 A 16,16 0,0,0 264,3056" class="nofill"></path>
    <path d="M 288,3040 A 16,16 0,0,1 288,3056" class="nofill"></path>
    <line x1="264" y1="3056" x2="266" y2="3060" class="solid"></line>
    <path d="M 266,3060 A 8,8 0,0,0 272,3064" class="nofill"></path>
    <line x1="272" y1="3064" x2="280" y2="3064" class="solid"></line>
    <line x1="288" y1="3056" x2="286" y2="3060" class="solid"></line>
    <path d="M 286,3060 A 8,8 0,0,1 280,3064" class="nofill"></path>
  </g>
  <g>
    <path d="M 40,3016 A 8,8 0,0,0 34,3020" class="nofill"></path>
    <line x1="34" y1="3020" x2="32" y2="3024" class="solid"></line>
    <line x1="40" y1="3016" x2="48" y2="3016" class="solid"></line>
    <path d="M 48,3016 A 8,8 0,0,1 54,3020" class="nofill"></path>
    <line x1="54" y1="3020" x2="56" y2="3024" class="solid"></line>
    <path d="M 32,3024 A 16,16 0,0,0 32,3040" class="nofill"></path>
    <path d="M 56,3024 A 16,16 0,0,1 56,3040" class="nofill"></path>
    <line x1="32" y1="3040" x2="34" y2="3044" class="solid"></line>
    <path d="M 34,3044 A 8,8 0,0,0 40,3048" class="nofill"></path>
    <line x1="40" y1="3048" x2="48" y2="3048" class="solid"></line>
    <line x1="56" y1="3040" x2="54" y2="3044" class="solid"></line>
    <path d="M 54,3044 A 8,8 0,0,1 48,3048" class="nofill"></path>
  </g>
  <g>
    <path d="M 104,3176 A 4,4 0,0,0 100,3180" class="nofill"></path>
    <line x1="100" y1="3180" x2="100" y2="3220" class="solid"></line>
    <line x1="104" y1="3176" x2="296" y2="3176" class="solid"></line>
    <line x1="68" y1="3224" x2="96" y2="3224" class="solid"></line>
    <path d="M 100,3220 A 4,4 0,0,1 96,3224" class="nofill"></path>
    <path d="M 100,3220 A 4,4 0,0,0 104,3224" class="nofill"></path>
    <line x1="104" y1="3224" x2="296" y2="3224" class="broken"></line>
  </g>
  <g>
    <path d="M 128,3272 A 8,8 0,0,0 122,3276" class="nofill"></path>
    <line x1="122" y1="3276" x2="102" y2="3316" class="solid"></line>
    <line x1="128" y1="3272" x2="296" y2="3272" class="solid"></line>
    <line x1="68" y1="3320" x2="96" y2="3320" class="solid"></line>
    <path d="M 102,3316 A 8,8 0,0,1 96,3320" class="nofill"></path>
    <path d="M 102,3316 A 3,3 0,0,0 104,3320" class="nofill"></path>
    <line x1="104" y1="3320" x2="136" y2="3320" class="broken"></line>
  </g>
  <g>
    <line x1="72" y1="3384" x2="200" y2="3384" class="broken"></line>
    <path d="M 200,3384 A 4,4 0,0,1 204,3388" class="nofill"></path>
    <path d="M 208,3384 A 4,4 0,0,0 204,3388" class="nofill"></path>
    <line x1="204" y1="3388" x2="204" y2="3428" class="solid"></line>
    <line x1="208" y1="3384" x2="296" y2="3384" class="solid"></line>
    <line x1="68" y1="3432" x2="200" y2="3432" class="solid"></line>
    <path d="M 204,3428 A 4,4 0,0,1 200,3432" class="nofill"></path>
    <path d="M 204,3428 A 4,4 0,0,0 208,3432" class="nofill"></path>
    <line x1="208" y1="3432" x2="296" y2="3432" class="broken"></line>
  </g>
  <g>
    <line x1="124" y1="3496" x2="180" y2="3496" class="solid"></line>
    <line x1="124" y1="3496" x2="124" y2="3528" class="solid"></line>
    <line x1="180" y1="3496" x2="180" y2="3528" class="solid"></line>
    <line x1="96" y1="3512" x2="124" y2="3512" class="solid"></line>
    <line x1="180" y1="3512" x2="200" y2="3512" class="solid"></line>
    <line x1="124" y1="3528" x2="180" y2="3528" class="solid"></line>
  </g>
  <g>
    <path d="M 216,3496 A 8,8 0,0,0 210,3500" class="nofill"></path>
    <line x1="210" y1="3500" x2="208" y2="3504" class="solid"></line>
    <line x1="216" y1="3496" x2="224" y2="3496" class="solid"></line>
    <path d="M 224,3496 A 8,8 0,0,1 230,3500" class="nofill"></path>
    <line x1="230" y1="3500" x2="232" y2="3504" class="solid"></line>
    <path d="M 208,3504 A 16,16 0,0,0 208,3520" class="nofill"></path>
    <path d="M 232,3504 A 16,16 0,0,1 232,3520" class="nofill"></path>
    <line x1="208" y1="3520" x2="210" y2="3524" class="solid"></line>
    <path d="M 210,3524 A 8,8 0,0,0 216,3528" class="nofill"></path>
    <line x1="216" y1="3528" x2="224" y2="3528" class="solid"></line>
    <line x1="232" y1="3520" x2="230" y2="3524" class="solid"></line>
    <path d="M 230,3524 A 8,8 0,0,1 224,3528" class="nofill"></path>
  </g>
  <g>
    <line x1="252" y1="3496" x2="284" y2="3496" class="solid"></line>
    <line x1="252" y1="3496" x2="252" y2="3528" class="solid"></line>
    <line x1="284" y1="3496" x2="284" y2="3528" class="solid"></line>
    <line x1="240" y1="3512" x2="252" y2="3512" class="solid"></line>
    <line x1="284" y1="3512" x2="304" y2="3512" class="solid"></line>
    <line x1="252" y1="3528" x2="284" y2="3528" class="solid"></line>
  </g>
  <g>
    <line x1="80" y1="3512" x2="88" y2="3512" class="solid"></line>
    <path d="M 88,3512 A 4,4 0,0,1 92,3516" class="nofill"></path>
    <line x1="92" y1="3516" x2="92" y2="3572" class="solid"></line>
    <path d="M 92,3572 A 4,4 0,0,0 96,3576" class="nofill"></path>
    <line x1="96" y1="3576" x2="104" y2="3576" class="solid"></line>
    <path d="M 112,3544 A 4,4 0,0,0 108,3548" class="nofill"></path>
    <line x1="108" y1="3548" x2="108" y2="3572" class="solid"></line>
    <line x1="112" y1="3544" x2="160" y2="3544" class="solid"></line>
    <path d="M 108,3572 A 4,4 0,0,1 104,3576" class="nofill"></path>
  </g>
  <g>
    <path d="M 312,3512 A 4,4 0,0,0 308,3516" class="nofill"></path>
    <line x1="308" y1="3516" x2="308" y2="3572" class="solid"></line>
    <line x1="168" y1="3544" x2="216" y2="3544" class="solid"></line>
    <path d="M 216,3544 A 4,4 0,0,1 220,3548" class="nofill"></path>
    <line x1="220" y1="3548" x2="220" y2="3572" class="solid"></line>
    <path d="M 220,3572 A 4,4 0,0,0 224,3576" class="nofill"></path>
    <line x1="224" y1="3576" x2="304" y2="3576" class="solid"></line>
    <path d="M 308,3572 A 4,4 0,0,1 304,3576" class="nofill"></path>
  </g>
  <g>
    <line x1="148" y1="3560" x2="180" y2="3560" class="solid"></line>
    <line x1="148" y1="3560" x2="148" y2="3592" class="solid"></line>
    <line x1="180" y1="3560" x2="180" y2="3592" class="solid"></line>
    <line x1="180" y1="3576" x2="200" y2="3576" class="solid"></line>
    <path d="M 200,3576 A 4,4 0,0,1 204,3580" class="nofill"></path>
    <line x1="204" y1="3580" x2="204" y2="3620" class="solid"></line>
    <line x1="148" y1="3592" x2="180" y2="3592" class="solid"></line>
    <path d="M 128,3576 A 4,4 0,0,0 124,3580" class="nofill"></path>
    <line x1="124" y1="3580" x2="124" y2="3620" class="solid"></line>
    <line x1="128" y1="3576" x2="148" y2="3576" class="solid"></line>
    <path d="M 124,3620 A 4,4 0,0,0 128,3624" class="nofill"></path>
    <line x1="128" y1="3624" x2="144" y2="3624" class="solid"></line>
    <line x1="184" y1="3624" x2="200" y2="3624" class="solid"></line>
    <path d="M 204,3620 A 4,4 0,0,1 200,3624" class="nofill"></path>
  </g>
  <g>
    <line x1="92" y1="3584" x2="92" y2="3684" class="solid"></line>
    <path d="M 92,3684 A 4,4 0,0,0 96,3688" class="nofill"></path>
    <line x1="96" y1="3688" x2="112" y2="3688" class="solid"></line>
  </g>
  <g>
    <path d="M 160,3608 A 8,8 0,0,0 154,3612" class="nofill"></path>
    <line x1="154" y1="3612" x2="152" y2="3616" class="solid"></line>
    <line x1="160" y1="3608" x2="168" y2="3608" class="solid"></line>
    <path d="M 168,3608 A 8,8 0,0,1 174,3612" class="nofill"></path>
    <line x1="174" y1="3612" x2="176" y2="3616" class="solid"></line>
    <path d="M 152,3616 A 16,16 0,0,0 152,3632" class="nofill"></path>
    <path d="M 176,3616 A 16,16 0,0,1 176,3632" class="nofill"></path>
    <line x1="152" y1="3632" x2="154" y2="3636" class="solid"></line>
    <path d="M 154,3636 A 8,8 0,0,0 160,3640" class="nofill"></path>
    <line x1="160" y1="3640" x2="168" y2="3640" class="solid"></line>
    <line x1="176" y1="3632" x2="174" y2="3636" class="solid"></line>
    <path d="M 174,3636 A 8,8 0,0,1 168,3640" class="nofill"></path>
  </g>
  <g>
    <line x1="308" y1="3628" x2="308" y2="3684" class="solid"></line>
    <line x1="272" y1="3688" x2="304" y2="3688" class="solid"></line>
    <path d="M 308,3684 A 4,4 0,0,1 304,3688" class="nofill"></path>
  </g>
  <g>
    <path d="M 120,3656 A 4,4 0,0,0 116,3660" class="nofill"></path>
    <line x1="116" y1="3660" x2="116" y2="3684" class="solid"></line>
    <line x1="120" y1="3656" x2="184" y2="3656" class="solid"></line>
    <path d="M 116,3684 A 4,4 0,0,0 120,3688" class="nofill"></path>
    <line x1="120" y1="3688" x2="128" y2="3688" class="solid"></line>
  </g>
  <g>
    <line x1="192" y1="3656" x2="264" y2="3656" class="solid"></line>
    <path d="M 264,3656 A 4,4 0,0,1 268,3660" class="nofill"></path>
    <line x1="268" y1="3660" x2="268" y2="3684" class="solid"></line>
    <line x1="256" y1="3688" x2="264" y2="3688" class="solid"></line>
    <path d="M 268,3684 A 4,4 0,0,1 264,3688" class="nofill"></path>
  </g>
  <g>
    <line x1="148" y1="3672" x2="180" y2="3672" class="solid"></line>
    <line x1="148" y1="3672" x2="148" y2="3704" class="solid"></line>
    <line x1="180" y1="3672" x2="180" y2="3704" class="solid"></line>
    <line x1="180" y1="3688" x2="200" y2="3688" class="solid"></line>
    <line x1="148" y1="3704" x2="180" y2="3704" class="solid"></line>
    <path d="M 136,3688 A 4,4 0,0,0 132,3692" class="nofill"></path>
    <line x1="132" y1="3692" x2="132" y2="3716" class="solid"></line>
    <line x1="136" y1="3688" x2="148" y2="3688" class="solid"></line>
    <path d="M 132,3716 A 4,4 0,0,0 136,3720" class="nofill"></path>
    <line x1="136" y1="3720" x2="192" y2="3720" class="solid"></line>
  </g>
  <g>
    <path d="M 216,3672 A 8,8 0,0,0 210,3676" class="nofill"></path>
    <line x1="210" y1="3676" x2="208" y2="3680" class="solid"></line>
    <line x1="216" y1="3672" x2="224" y2="3672" class="solid"></line>
    <path d="M 224,3672 A 8,8 0,0,1 230,3676" class="nofill"></path>
    <line x1="230" y1="3676" x2="232" y2="3680" class="solid"></line>
    <path d="M 208,3680 A 16,16 0,0,0 208,3696" class="nofill"></path>
    <path d="M 232,3680 A 16,16 0,0,1 232,3696" class="nofill"></path>
    <line x1="208" y1="3696" x2="210" y2="3700" class="solid"></line>
    <path d="M 210,3700 A 8,8 0,0,0 216,3704" class="nofill"></path>
    <line x1="216" y1="3704" x2="224" y2="3704" class="solid"></line>
    <line x1="232" y1="3696" x2="230" y2="3700" class="solid"></line>
    <path d="M 230,3700 A 8,8 0,0,1 224,3704" class="nofill"></path>
  </g>
  <g>
    <line x1="240" y1="3688" x2="248" y2="3688" class="solid"></line>
    <path d="M 248,3688 A 4,4 0,0,1 252,3692" class="nofill"></path>
    <line x1="252" y1="3692" x2="252" y2="3716" class="solid"></line>
    <line x1="200" y1="3720" x2="248" y2="3720" class="solid"></line>
    <path d="M 252,3716 A 4,4 0,0,1 248,3720" class="nofill"></path>
  </g>
  <g>
    <path d="M 320,3736 A 4,4 0,0,0 316,3740" class="nofill"></path>
    <line x1="316" y1="3740" x2="316" y2="3768" class="solid"></line>
    <line x1="320" y1="3736" x2="416" y2="3736" class="solid"></line>
    <line x1="304" y1="3776" x2="312" y2="3776" class="solid"></line>
    <path d="M 316,3768 A 8,8 0,0,1 312,3776" class="nofill"></path>
  </g>
  <g>
    <line x1="424" y1="3736" x2="544" y2="3736" class="solid"></line>
    <path d="M 544,3736 A 4,4 0,0,1 548,3740" class="nofill"></path>
    <line x1="548" y1="3740" x2="548" y2="3768" class="solid"></line>
    <path d="M 548,3768 A 8,8 0,0,0 552,3776" class="nofill"></path>
    <line x1="552" y1="3776" x2="576" y2="3776" class="solid"></line>
  </g>
  <g>
    <line x1="92" y1="3752" x2="204" y2="3752" class="solid"></line>
    <line x1="92" y1="3752" x2="92" y2="3784" class="solid"></line>
    <line x1="204" y1="3752" x2="204" y2="3784" class="solid"></line>
    <line x1="56" y1="3776" x2="92" y2="3776" class="solid"></line>
    <line x1="204" y1="3776" x2="216" y2="3776" class="solid"></line>
    <line x1="92" y1="3784" x2="204" y2="3784" class="solid"></line>
  </g>
  <g>
    <path d="M 232,3752 A 8,8 0,0,0 226,3756" class="nofill"></path>
    <line x1="226" y1="3756" x2="224" y2="3760" class="solid"></line>
    <line x1="232" y1="3752" x2="240" y2="3752" class="solid"></line>
    <path d="M 240,3752 A 8,8 0,0,1 246,3756" class="nofill"></path>
    <line x1="246" y1="3756" x2="248" y2="3760" class="solid"></line>
    <path d="M 224,3760 A 16,16 0,0,0 224,3776" class="nofill"></path>
    <path d="M 248,3760 A 16,16 0,0,1 248,3776" class="nofill"></path>
    <line x1="224" y1="3776" x2="226" y2="3780" class="solid"></line>
    <path d="M 226,3780 A 8,8 0,0,0 232,3784" class="nofill"></path>
    <line x1="232" y1="3784" x2="240" y2="3784" class="solid"></line>
    <line x1="248" y1="3776" x2="246" y2="3780" class="solid"></line>
    <path d="M 246,3780 A 8,8 0,0,1 240,3784" class="nofill"></path>
  </g>
  <g>
    <path d="M 280,3752 A 8,8 0,0,0 274,3756" class="nofill"></path>
    <line x1="274" y1="3756" x2="272" y2="3760" class="solid"></line>
    <line x1="280" y1="3752" x2="288" y2="3752" class="solid"></line>
    <path d="M 288,3752 A 8,8 0,0,1 294,3756" class="nofill"></path>
    <line x1="294" y1="3756" x2="296" y2="3760" class="solid"></line>
    <path d="M 272,3760 A 16,16 0,0,0 272,3776" class="nofill"></path>
    <path d="M 296,3760 A 16,16 0,0,1 296,3776" class="nofill"></path>
    <line x1="272" y1="3776" x2="274" y2="3780" class="solid"></line>
    <path d="M 274,3780 A 8,8 0,0,0 280,3784" class="nofill"></path>
    <line x1="280" y1="3784" x2="288" y2="3784" class="solid"></line>
    <line x1="296" y1="3776" x2="294" y2="3780" class="solid"></line>
    <path d="M 294,3780 A 8,8 0,0,1 288,3784" class="nofill"></path>
  </g>
  <g>
    <line x1="340" y1="3752" x2="396" y2="3752" class="solid"></line>
    <line x1="340" y1="3752" x2="340" y2="3784" class="solid"></line>
    <line x1="396" y1="3752" x2="396" y2="3784" class="solid"></line>
    <line x1="320" y1="3776" x2="340" y2="3776" class="solid"></line>
    <line x1="396" y1="3776" x2="408" y2="3776" class="solid"></line>
    <line x1="340" y1="3784" x2="396" y2="3784" class="solid"></line>
  </g>
  <g>
    <path d="M 424,3752 A 8,8 0,0,0 418,3756" class="nofill"></path>
    <line x1="418" y1="3756" x2="416" y2="3760" class="solid"></line>
    <line x1="424" y1="3752" x2="432" y2="3752" class="solid"></line>
    <path d="M 432,3752 A 8,8 0,0,1 438,3756" class="nofill"></path>
    <line x1="438" y1="3756" x2="440" y2="3760" class="solid"></line>
    <path d="M 416,3760 A 16,16 0,0,0 416,3776" class="nofill"></path>
    <path d="M 440,3760 A 16,16 0,0,1 440,3776" class="nofill"></path>
    <line x1="416" y1="3776" x2="418" y2="3780" class="solid"></line>
    <path d="M 418,3780 A 8,8 0,0,0 424,3784" class="nofill"></path>
    <line x1="424" y1="3784" x2="432" y2="3784" class="solid"></line>
    <line x1="440" y1="3776" x2="438" y2="3780" class="solid"></line>
    <path d="M 438,3780 A 8,8 0,0,1 432,3784" class="nofill"></path>
  </g>
  <g>
    <line x1="468" y1="3752" x2="516" y2="3752" class="solid"></line>
    <line x1="468" y1="3752" x2="468" y2="3784" class="solid"></line>
    <line x1="516" y1="3752" x2="516" y2="3784" class="solid"></line>
    <line x1="448" y1="3776" x2="468" y2="3776" class="solid"></line>
    <line x1="516" y1="3776" x2="544" y2="3776" class="solid"></line>
    <line x1="468" y1="3784" x2="516" y2="3784" class="solid"></line>
  </g>
  <g>
    <path d="M 592,3752 A 8,8 0,0,0 586,3756" class="nofill"></path>
    <line x1="586" y1="3756" x2="584" y2="3760" class="solid"></line>
    <line x1="592" y1="3752" x2="600" y2="3752" class="solid"></line>
    <path d="M 600,3752 A 8,8 0,0,1 606,3756" class="nofill"></path>
    <line x1="606" y1="3756" x2="608" y2="3760" class="solid"></line>
    <path d="M 584,3760 A 16,16 0,0,0 584,3776" class="nofill"></path>
    <path d="M 608,3760 A 16,16 0,0,1 608,3776" class="nofill"></path>
    <line x1="584" y1="3776" x2="586" y2="3780" class="solid"></line>
    <path d="M 586,3780 A 8,8 0,0,0 592,3784" class="nofill"></path>
    <line x1="592" y1="3784" x2="600" y2="3784" class="solid"></line>
    <line x1="608" y1="3776" x2="606" y2="3780" class="solid"></line>
    <path d="M 606,3780 A 8,8 0,0,1 600,3784" class="nofill"></path>
  </g>
  <g>
    <line x1="636" y1="3752" x2="692" y2="3752" class="solid"></line>
    <line x1="636" y1="3752" x2="636" y2="3784" class="solid"></line>
    <line x1="692" y1="3752" x2="692" y2="3784" class="solid"></line>
    <line x1="616" y1="3776" x2="636" y2="3776" class="solid"></line>
    <line x1="692" y1="3776" x2="744" y2="3776" class="solid"></line>
    <line x1="636" y1="3784" x2="692" y2="3784" class="solid"></line>
  </g>
  <g>
    <path d="M 64,3776 A 8,8 0,0,1 68,3784" class="nofill"></path>
    <line x1="68" y1="3784" x2="68" y2="3844" class="solid"></line>
    <path d="M 68,3844 A 4,4 0,0,0 72,3848" class="nofill"></path>
    <line x1="72" y1="3848" x2="712" y2="3848" class="solid"></line>
    <path d="M 720,3776 A 8,8 0,0,0 716,3784" class="nofill"></path>
    <line x1="716" y1="3784" x2="716" y2="3844" class="solid"></line>
    <path d="M 716,3844 A 4,4 0,0,1 712,3848" class="nofill"></path>
  </g>
  <g>
    <path d="M 328,3776 A 8,8 0,0,0 324,3784" class="nofill"></path>
    <line x1="324" y1="3784" x2="324" y2="3812" class="solid"></line>
    <path d="M 324,3812 A 4,4 0,0,0 328,3816" class="nofill"></path>
    <line x1="328" y1="3816" x2="424" y2="3816" class="solid"></line>
  </g>
  <g>
    <path d="M 536,3776 A 8,8 0,0,1 540,3784" class="nofill"></path>
    <line x1="540" y1="3784" x2="540" y2="3812" class="solid"></line>
    <line x1="520" y1="3816" x2="536" y2="3816" class="solid"></line>
    <path d="M 540,3812 A 4,4 0,0,1 536,3816" class="nofill"></path>
  </g>
  <g>
    <path d="M 496,3800 A 8,8 0,0,0 490,3804" class="nofill"></path>
    <line x1="490" y1="3804" x2="488" y2="3808" class="solid"></line>
    <line x1="496" y1="3800" x2="504" y2="3800" class="solid"></line>
    <path d="M 504,3800 A 8,8 0,0,1 510,3804" class="nofill"></path>
    <line x1="510" y1="3804" x2="512" y2="3808" class="solid"></line>
    <path d="M 488,3808 A 16,16 0,0,0 488,3824" class="nofill"></path>
    <path d="M 512,3808 A 16,16 0,0,1 512,3824" class="nofill"></path>
    <line x1="488" y1="3824" x2="490" y2="3828" class="solid"></line>
    <path d="M 490,3828 A 8,8 0,0,0 496,3832" class="nofill"></path>
    <line x1="496" y1="3832" x2="504" y2="3832" class="solid"></line>
    <line x1="512" y1="3824" x2="510" y2="3828" class="solid"></line>
    <path d="M 510,3828 A 8,8 0,0,1 504,3832" class="nofill"></path>
  </g>
  <g>
    <line x1="256" y1="3928" x2="284" y2="3928" class="solid"></line>
    <line x1="284" y1="3928" x2="284" y2="3936" class="solid"></line>
  </g>
  <g>
    <line x1="260" y1="3932" x2="260" y2="3992" class="solid"></line>
    <line x1="212" y1="3948" x2="212" y2="3992" class="solid"></line>
    <line x1="236" y1="3948" x2="236" y2="3992" class="solid"></line>
    <line x1="284" y1="3948" x2="284" y2="3992" class="solid"></line>
    <line x1="164" y1="3964" x2="164" y2="3992" class="solid"></line>
    <line x1="188" y1="3964" x2="188" y2="3992" class="solid"></line>
    <line x1="88" y1="3992" x2="284" y2="3992" class="solid"></line>
    <line x1="116" y1="3984" x2="116" y2="3992" class="solid"></line>
    <line x1="140" y1="3984" x2="140" y2="3992" class="solid"></line>
  </g>
  <g>
    <line x1="580" y1="3952" x2="580" y2="4000" class="solid"></line>
    <line x1="392" y1="3992" x2="580" y2="3992" class="solid"></line>
  </g>
  <g>
    <line x1="68" y1="4064" x2="68" y2="4248" class="solid"></line>
    <line x1="64" y1="4072" x2="68" y2="4072" class="solid"></line>
    <line x1="64" y1="4088" x2="68" y2="4088" class="solid"></line>
    <line x1="64" y1="4104" x2="68" y2="4104" class="solid"></line>
    <line x1="64" y1="4120" x2="68" y2="4120" class="solid"></line>
    <line x1="64" y1="4136" x2="68" y2="4136" class="solid"></line>
    <line x1="64" y1="4152" x2="68" y2="4152" class="solid"></line>
    <line x1="64" y1="4168" x2="68" y2="4168" class="solid"></line>
    <line x1="64" y1="4184" x2="68" y2="4184" class="solid"></line>
    <line x1="64" y1="4200" x2="68" y2="4200" class="solid"></line>
    <line x1="64" y1="4232" x2="68" y2="4232" class="solid"></line>
    <line x1="68" y1="4248" x2="584" y2="4248" class="solid"></line>
    <line x1="164" y1="4240" x2="164" y2="4248" class="solid"></line>
    <line x1="260" y1="4240" x2="260" y2="4248" class="solid"></line>
    <line x1="356" y1="4240" x2="356" y2="4248" class="solid"></line>
    <line x1="452" y1="4240" x2="452" y2="4248" class="solid"></line>
    <line x1="548" y1="4240" x2="548" y2="4248" class="solid"></line>
  </g>
  <g>
    <path d="M 392,4072 A 4,4 0,0,0 388,4076" class="nofill"></path>
    <line x1="388" y1="4076" x2="388" y2="4132" class="solid"></line>
    <path d="M 392,4072 A 4,4 0,0,1 396,4076" class="nofill"></path>
    <line x1="396" y1="4076" x2="396" y2="4132" class="solid"></line>
    <path d="M 396,4132 A 4,4 0,0,0 400,4136" class="nofill"></path>
    <line x1="400" y1="4136" x2="408" y2="4136" class="solid"></line>
    <path d="M 360,4120 A 4,4 0,0,0 356,4124" class="nofill"></path>
    <line x1="356" y1="4124" x2="356" y2="4164" class="solid"></line>
    <line x1="360" y1="4120" x2="368" y2="4120" class="solid"></line>
    <path d="M 368,4120 A 4,4 0,0,1 372,4124" class="nofill"></path>
    <line x1="372" y1="4124" x2="372" y2="4132" class="solid"></line>
    <path d="M 372,4132 A 4,4 0,0,0 376,4136" class="nofill"></path>
    <line x1="376" y1="4136" x2="384" y2="4136" class="solid"></line>
    <path d="M 388,4132 A 4,4 0,0,1 384,4136" class="nofill"></path>
    <path d="M 424,4104 A 4,4 0,0,0 420,4108" class="nofill"></path>
    <line x1="420" y1="4108" x2="420" y2="4116" class="solid"></line>
    <path d="M 424,4104 A 4,4 0,0,1 428,4108" class="nofill"></path>
    <line x1="428" y1="4108" x2="428" y2="4116" class="solid"></line>
    <path d="M 428,4116 A 4,4 0,0,0 432,4120" class="nofill"></path>
    <path d="M 432,4120 A 4,4 0,0,1 436,4124" class="nofill"></path>
    <line x1="436" y1="4124" x2="436" y2="4148" class="solid"></line>
    <path d="M 436,4148 A 4,4 0,0,0 440,4152" class="nofill"></path>
    <path d="M 448,4104 A 4,4 0,0,0 444,4108" class="nofill"></path>
    <line x1="444" y1="4108" x2="444" y2="4148" class="solid"></line>
    <path d="M 448,4104 A 4,4 0,0,1 452,4108" class="nofill"></path>
    <line x1="452" y1="4108" x2="452" y2="4116" class="solid"></line>
    <path d="M 452,4116 A 4,4 0,0,0 456,4120" class="nofill"></path>
    <line x1="456" y1="4120" x2="464" y2="4120" class="solid"></line>
    <path d="M 444,4148 A 4,4 0,0,1 440,4152" class="nofill"></path>
    <path d="M 472,4104 A 4,4 0,0,0 468,4108" class="nofill"></path>
    <line x1="468" y1="4108" x2="468" y2="4116" class="solid"></line>
    <path d="M 472,4104 A 4,4 0,0,1 476,4108" class="nofill"></path>
    <line x1="476" y1="4108" x2="476" y2="4116" class="solid"></line>
    <path d="M 468,4116 A 4,4 0,0,1 464,4120" class="nofill"></path>
    <path d="M 476,4116 A 4,4 0,0,0 480,4120" class="nofill"></path>
    <path d="M 480,4120 A 4,4 0,0,1 484,4124" class="nofill"></path>
    <line x1="484" y1="4124" x2="484" y2="4148" class="solid"></line>
    <path d="M 484,4148 A 4,4 0,0,0 488,4152" class="nofill"></path>
    <path d="M 416,4120 A 4,4 0,0,0 412,4124" class="nofill"></path>
    <line x1="412" y1="4124" x2="412" y2="4132" class="solid"></line>
    <path d="M 420,4116 A 4,4 0,0,1 416,4120" class="nofill"></path>
    <path d="M 412,4132 A 4,4 0,0,1 408,4136" class="nofill"></path>
    <path d="M 272,4120 A 4,4 0,0,0 268,4124" class="nofill"></path>
    <line x1="268" y1="4124" x2="268" y2="4164" class="solid"></line>
    <path d="M 272,4120 A 4,4 0,0,1 276,4124" class="nofill"></path>
    <line x1="276" y1="4124" x2="276" y2="4164" class="solid"></line>
    <path d="M 276,4164 A 4,4 0,0,0 280,4168" class="nofill"></path>
    <path d="M 256,4136 A 4,4 0,0,0 252,4140" class="nofill"></path>
    <line x1="252" y1="4140" x2="252" y2="4164" class="solid"></line>
    <path d="M 256,4136 A 4,4 0,0,1 260,4140" class="nofill"></path>
    <line x1="260" y1="4140" x2="260" y2="4164" class="solid"></line>
    <path d="M 260,4164 A 4,4 0,0,0 264,4168" class="nofill"></path>
    <path d="M 268,4164 A 4,4 0,0,1 264,4168" class="nofill"></path>
    <path d="M 288,4136 A 4,4 0,0,0 284,4140" class="nofill"></path>
    <line x1="284" y1="4140" x2="284" y2="4164" class="solid"></line>
    <path d="M 288,4136 A 4,4 0,0,1 292,4140" class="nofill"></path>
    <line x1="292" y1="4140" x2="292" y2="4148" class="solid"></line>
    <path d="M 292,4148 A 4,4 0,0,0 296,4152" class="nofill"></path>
    <path d="M 296,4152 A 4,4 0,0,1 300,4156" class="nofill"></path>
    <line x1="300" y1="4156" x2="300" y2="4164" class="solid"></line>
    <path d="M 284,4164 A 4,4 0,0,1 280,4168" class="nofill"></path>
    <path d="M 300,4164 A 4,4 0,0,0 304,4168" class="nofill"></path>
    <path d="M 312,4136 A 4,4 0,0,0 308,4140" class="nofill"></path>
    <line x1="308" y1="4140" x2="308" y2="4164" class="solid"></line>
    <path d="M 312,4136 A 4,4 0,0,1 316,4140" class="nofill"></path>
    <line x1="316" y1="4140" x2="316" y2="4148" class="solid"></line>
    <path d="M 316,4148 A 4,4 0,0,0 320,4152" class="nofill"></path>
    <path d="M 308,4164 A 4,4 0,0,1 304,4168" class="nofill"></path>
    <path d="M 328,4136 A 4,4 0,0,0 324,4140" class="nofill"></path>
    <line x1="324" y1="4140" x2="324" y2="4148" class="solid"></line>
    <path d="M 328,4136 A 4,4 0,0,1 332,4140" class="nofill"></path>
    <line x1="332" y1="4140" x2="332" y2="4164" class="solid"></line>
    <path d="M 324,4148 A 4,4 0,0,1 320,4152" class="nofill"></path>
    <path d="M 332,4164 A 4,4 0,0,0 336,4168" class="nofill"></path>
    <line x1="336" y1="4168" x2="352" y2="4168" class="solid"></line>
    <path d="M 356,4164 A 4,4 0,0,1 352,4168" class="nofill"></path>
    <path d="M 552,4088 A 4,4 0,0,0 548,4092" class="nofill"></path>
    <line x1="548" y1="4092" x2="548" y2="4100" class="solid"></line>
    <path d="M 552,4088 A 4,4 0,0,1 556,4092" class="nofill"></path>
    <line x1="556" y1="4092" x2="556" y2="4100" class="solid"></line>
    <path d="M 556,4100 A 4,4 0,0,0 560,4104" class="nofill"></path>
    <line x1="560" y1="4104" x2="568" y2="4104" class="solid"></line>
    <path d="M 568,4104 A 4,4 0,0,1 572,4108" class="nofill"></path>
    <line x1="572" y1="4108" x2="572" y2="4132" class="solid"></line>
    <path d="M 572,4132 A 4,4 0,0,0 576,4136" class="nofill"></path>
    <path d="M 584,4120 A 4,4 0,0,0 580,4124" class="nofill"></path>
    <line x1="580" y1="4124" x2="580" y2="4132" class="solid"></line>
    <path d="M 580,4132 A 4,4 0,0,1 576,4136" class="nofill"></path>
    <path d="M 512,4104 A 4,4 0,0,0 508,4108" class="nofill"></path>
    <line x1="508" y1="4108" x2="508" y2="4148" class="solid"></line>
    <path d="M 512,4104 A 4,4 0,0,1 516,4108" class="nofill"></path>
    <line x1="516" y1="4108" x2="516" y2="4116" class="solid"></line>
    <path d="M 516,4116 A 4,4 0,0,0 520,4120" class="nofill"></path>
    <line x1="520" y1="4120" x2="536" y2="4120" class="solid"></line>
    <path d="M 544,4104 A 4,4 0,0,0 540,4108" class="nofill"></path>
    <line x1="540" y1="4108" x2="540" y2="4116" class="solid"></line>
    <path d="M 548,4100 A 4,4 0,0,1 544,4104" class="nofill"></path>
    <path d="M 540,4116 A 4,4 0,0,1 536,4120" class="nofill"></path>
    <path d="M 496,4120 A 4,4 0,0,0 492,4124" class="nofill"></path>
    <line x1="492" y1="4124" x2="492" y2="4148" class="solid"></line>
    <path d="M 496,4120 A 4,4 0,0,1 500,4124" class="nofill"></path>
    <line x1="500" y1="4124" x2="500" y2="4148" class="solid"></line>
    <path d="M 492,4148 A 4,4 0,0,1 488,4152" class="nofill"></path>
    <path d="M 500,4148 A 4,4 0,0,0 504,4152" class="nofill"></path>
    <path d="M 508,4148 A 4,4 0,0,1 504,4152" class="nofill"></path>
    <path d="M 104,4136 A 4,4 0,0,0 100,4140" class="nofill"></path>
    <line x1="100" y1="4140" x2="100" y2="4196" class="solid"></line>
    <path d="M 104,4136 A 4,4 0,0,1 108,4140" class="nofill"></path>
    <line x1="108" y1="4140" x2="108" y2="4148" class="solid"></line>
    <path d="M 108,4148 A 4,4 0,0,0 112,4152" class="nofill"></path>
    <path d="M 112,4152 A 4,4 0,0,1 116,4156" class="nofill"></path>
    <line x1="116" y1="4156" x2="116" y2="4164" class="solid"></line>
    <path d="M 116,4164 A 4,4 0,0,0 120,4168" class="nofill"></path>
    <path d="M 120,4168 A 4,4 0,0,1 124,4172" class="nofill"></path>
    <line x1="124" y1="4172" x2="124" y2="4180" class="solid"></line>
    <path d="M 124,4180 A 4,4 0,0,0 128,4184" class="nofill"></path>
    <path d="M 136,4168 A 4,4 0,0,0 132,4172" class="nofill"></path>
    <line x1="132" y1="4172" x2="132" y2="4180" class="solid"></line>
    <path d="M 136,4168 A 4,4 0,0,1 140,4172" class="nofill"></path>
    <line x1="140" y1="4172" x2="140" y2="4180" class="solid"></line>
    <path d="M 132,4180 A 4,4 0,0,1 128,4184" class="nofill"></path>
    <path d="M 140,4180 A 4,4 0,0,0 144,4184" class="nofill"></path>
    <path d="M 144,4184 A 4,4 0,0,1 148,4188" class="nofill"></path>
    <line x1="148" y1="4188" x2="148" y2="4196" class="solid"></line>
    <path d="M 148,4196 A 4,4 0,0,0 152,4200" class="nofill"></path>
    <path d="M 176,4152 A 4,4 0,0,0 172,4156" class="nofill"></path>
    <line x1="172" y1="4156" x2="172" y2="4164" class="solid"></line>
    <line x1="176" y1="4152" x2="192" y2="4152" class="solid"></line>
    <path d="M 192,4152 A 4,4 0,0,1 196,4156" class="nofill"></path>
    <line x1="196" y1="4156" x2="196" y2="4164" class="solid"></line>
    <path d="M 196,4164 A 4,4 0,0,0 200,4168" class="nofill"></path>
    <line x1="200" y1="4168" x2="208" y2="4168" class="solid"></line>
    <path d="M 208,4168 A 4,4 0,0,1 212,4172" class="nofill"></path>
    <line x1="212" y1="4172" x2="212" y2="4196" class="solid"></line>
    <path d="M 212,4196 A 4,4 0,0,0 216,4200" class="nofill"></path>
    <path d="M 168,4168 A 4,4 0,0,0 164,4172" class="nofill"></path>
    <line x1="164" y1="4172" x2="164" y2="4180" class="solid"></line>
    <path d="M 172,4164 A 4,4 0,0,1 168,4168" class="nofill"></path>
    <path d="M 224,4168 A 4,4 0,0,0 220,4172" class="nofill"></path>
    <line x1="220" y1="4172" x2="220" y2="4196" class="solid"></line>
    <path d="M 224,4168 A 4,4 0,0,1 228,4172" class="nofill"></path>
    <line x1="228" y1="4172" x2="228" y2="4180" class="solid"></line>
    <path d="M 228,4180 A 4,4 0,0,0 232,4184" class="nofill"></path>
    <path d="M 220,4196 A 4,4 0,0,1 216,4200" class="nofill"></path>
    <path d="M 240,4168 A 4,4 0,0,0 236,4172" class="nofill"></path>
    <line x1="236" y1="4172" x2="236" y2="4180" class="solid"></line>
    <line x1="240" y1="4168" x2="248" y2="4168" class="solid"></line>
    <path d="M 252,4164 A 4,4 0,0,1 248,4168" class="nofill"></path>
    <path d="M 236,4180 A 4,4 0,0,1 232,4184" class="nofill"></path>
    <path d="M 160,4184 A 4,4 0,0,0 156,4188" class="nofill"></path>
    <line x1="156" y1="4188" x2="156" y2="4196" class="solid"></line>
    <path d="M 164,4180 A 4,4 0,0,1 160,4184" class="nofill"></path>
    <path d="M 156,4196 A 4,4 0,0,1 152,4200" class="nofill"></path>
    <path d="M 80,4168 A 4,4 0,0,0 76,4172" class="nofill"></path>
    <line x1="76" y1="4172" x2="76" y2="4212" class="solid"></line>
    <path d="M 80,4168 A 4,4 0,0,1 84,4172" class="nofill"></path>
    <line x1="84" y1="4172" x2="84" y2="4228" class="solid"></line>
    <path d="M 84,4228 A 4,4 0,0,0 88,4232" class="nofill"></path>
    <path d="M 96,4200 A 4,4 0,0,0 92,4204" class="nofill"></path>
    <line x1="92" y1="4204" x2="92" y2="4228" class="solid"></line>
    <path d="M 100,4196 A 4,4 0,0,1 96,4200" class="nofill"></path>
    <path d="M 92,4228 A 4,4 0,0,1 88,4232" class="nofill"></path>
    <line x1="64" y1="4216" x2="72" y2="4216" class="solid"></line>
    <path d="M 76,4212 A 4,4 0,0,1 72,4216" class="nofill"></path>
  </g>
  <g>
    <path d="M 184,4408 A 4,4 0,0,0 180,4412" class="nofill"></path>
    <line x1="180" y1="4412" x2="180" y2="4436" class="solid"></line>
    <line x1="184" y1="4408" x2="200" y2="4408" class="solid"></line>
    <path d="M 200,4408 A 4,4 0,0,1 204,4412" class="nofill"></path>
    <line x1="204" y1="4412" x2="204" y2="4528" class="solid"></line>
    <path d="M 80,4472 A 4,4 0,0,0 76,4476" class="nofill"></path>
    <line x1="76" y1="4476" x2="76" y2="4528" class="solid"></line>
    <line x1="80" y1="4472" x2="96" y2="4472" class="solid"></line>
    <line x1="148" y1="4504" x2="136" y2="4528" class="solid"></line>
    <line x1="148" y1="4504" x2="160" y2="4528" class="solid"></line>
    <line x1="76" y1="4528" x2="136" y2="4528" class="solid"></line>
    <line x1="160" y1="4528" x2="204" y2="4528" class="solid"></line>
    <line x1="136" y1="4528" x2="148" y2="4552" class="solid"></line>
    <line x1="160" y1="4528" x2="148" y2="4552" class="solid"></line>
    <line x1="148" y1="4552" x2="148" y2="4564" class="solid"></line>
    <path d="M 148,4564 A 4,4 0,0,0 152,4568" class="nofill"></path>
    <line x1="152" y1="4568" x2="168" y2="4568" class="solid"></line>
  </g>
  <g>
    <line x1="148" y1="4480" x2="148" y2="4500" class="solid"></line>
    <path d="M 148,4500 A 16,16 0,0,1 146,4508" class="nofill"></path>
    <path d="M 148,4500 A 16,16 0,0,0 150,4508" class="nofill"></path>
  </g>
  <g>
    <line x1="428" y1="4432" x2="428" y2="4564" class="solid"></line>
    <line x1="392" y1="4568" x2="424" y2="4568" class="solid"></line>
    <path d="M 428,4564 A 4,4 0,0,1 424,4568" class="nofill"></path>
  </g>
  <g>
    <line x1="192" y1="4560" x2="224" y2="4560" class="solid"></line>
    <line x1="192" y1="4560" x2="184" y2="4576" class="solid"></line>
    <line x1="184" y1="4576" x2="216" y2="4576" class="solid"></line>
    <line x1="224" y1="4560" x2="216" y2="4576" class="solid"></line>
  </g>
  <g>
    <line x1="84" y1="4784" x2="100" y2="4784" class="solid"></line>
    <line x1="84" y1="4784" x2="84" y2="4792" class="solid"></line>
    <line x1="84" y1="4792" x2="72" y2="4816" class="solid"></line>
    <line x1="100" y1="4784" x2="100" y2="4816" class="solid"></line>
    <line x1="84" y1="4800" x2="84" y2="4816" class="solid"></line>
    <line x1="84" y1="4816" x2="100" y2="4816" class="solid"></line>
  </g>
  <g>
    <text x="50" y="4812">⠶</text>
    <text x="58" y="4812">⠶</text>
  </g>
  <g>
    <path d="M 200,4760 A 8,8 0,0,0 194,4764" class="nofill"></path>
    <line x1="194" y1="4764" x2="192" y2="4768" class="solid"></line>
    <line x1="200" y1="4760" x2="208" y2="4760" class="solid"></line>
    <path d="M 192,4768 A 16,16 0,0,0 192,4784" class="nofill"></path>
  </g>
  <g>
    <path d="M 176,4776 A 8,8 0,0,0 170,4780" class="nofill"></path>
    <line x1="170" y1="4780" x2="168" y2="4784" class="solid"></line>
    <line x1="176" y1="4776" x2="184" y2="4776" class="solid"></line>
    <path d="M 168,4784 A 16,16 0,0,0 168,4800" class="nofill"></path>
    <line x1="168" y1="4800" x2="170" y2="4804" class="solid"></line>
    <path d="M 170,4804 A 8,8 0,0,0 176,4808" class="nofill"></path>
    <line x1="176" y1="4808" x2="184" y2="4808" class="solid"></line>
  </g>
  <g>
    <line x1="256" y1="4760" x2="264" y2="4760" class="solid"></line>
    <path d="M 264,4760 A 8,8 0,0,1 270,4764" class="nofill"></path>
    <line x1="270" y1="4764" x2="272" y2="4768" class="solid"></line>
    <path d="M 272,4768 A 16,16 0,0,1 272,4784" class="nofill"></path>
  </g>
  <g>
    <line x1="280" y1="4776" x2="288" y2="4776" class="solid"></line>
    <path d="M 288,4776 A 8,8 0,0,1 294,4780" class="nofill"></path>
    <line x1="294" y1="4780" x2="296" y2="4784" class="solid"></line>
  </g>
  <g>
    <line x1="524" y1="4800" x2="564" y2="4800" class="solid"></line>
    <line x1="524" y1="4800" x2="524" y2="4832" class="solid"></line>
    <line x1="564" y1="4800" x2="564" y2="4832" class="solid"></line>
    <line x1="524" y1="4832" x2="564" y2="4832" class="solid"></line>
    <line x1="528" y1="4832" x2="520" y2="4848" class="solid"></line>
  </g>
  <g>
    <path d="M 272,4800 A 16,16 0,0,1 272,4816" class="nofill"></path>
    <line x1="272" y1="4816" x2="270" y2="4820" class="solid"></line>
    <line x1="256" y1="4824" x2="264" y2="4824" class="solid"></line>
    <path d="M 270,4820 A 8,8 0,0,1 264,4824" class="nofill"></path>
  </g>
  <g>
    <text x="426" y="4812">...°</text>
    <text x="458" y="4812">]</text>
  </g>
  <g>
    <text x="530" y="4844">⠶</text>
    <text x="538" y="4844">⠶</text>
    <text x="546" y="4844">⠶</text>
  </g>
  <g>
    <line x1="320" y1="4888" x2="384" y2="4888" class="solid"></line>
    <path d="M 384,4888 A 3,3 0,0,1 386,4892" class="nofill"></path>
    <line x1="386" y1="4892" x2="374" y2="4916" class="solid"></line>
  </g>
  <g>
    <line x1="248" y1="4944" x2="320" y2="4944" class="solid"></line>
    <line x1="328" y1="4936" x2="320" y2="4944" class="solid"></line>
    <line x1="328" y1="4936" x2="476" y2="4936" class="solid"></line>
    <line x1="476" y1="4936" x2="464" y2="4960" class="solid"></line>
    <line x1="464" y1="4960" x2="476" y2="4984" class="solid"></line>
    <line x1="424" y1="4984" x2="476" y2="4984" class="solid"></line>
    <path d="M 168,4952 A 8,8 0,0,0 162,4956" class="nofill"></path>
    <line x1="162" y1="4956" x2="152" y2="4976" class="solid"></line>
    <line x1="168" y1="4952" x2="240" y2="4952" class="solid"></line>
    <line x1="248" y1="4944" x2="240" y2="4952" class="solid"></line>
    <path d="M 152,4976 A 16,16 0,0,0 152,4992" class="nofill"></path>
    <line x1="152" y1="4992" x2="162" y2="5012" class="solid"></line>
    <path d="M 162,5012 A 8,8 0,0,0 168,5016" class="nofill"></path>
    <line x1="168" y1="5016" x2="216" y2="5016" class="solid"></line>
    <line x1="216" y1="5016" x2="224" y2="5024" class="solid"></line>
    <line x1="224" y1="5024" x2="236" y2="5024" class="solid"></line>
    <line x1="236" y1="5024" x2="236" y2="5120" class="solid"></line>
    <line x1="200" y1="5120" x2="236" y2="5120" class="solid"></line>
    <line x1="200" y1="5120" x2="196" y2="5128" class="solid"></line>
    <line x1="196" y1="5128" x2="196" y2="5168" class="solid"></line>
    <path d="M 168,5160 A 4,4 0,0,0 164,5164" class="nofill"></path>
    <line x1="164" y1="5164" x2="164" y2="5184" class="solid"></line>
    <line x1="168" y1="5160" x2="196" y2="5160" class="solid"></line>
    <line x1="136" y1="5184" x2="164" y2="5184" class="solid"></line>
  </g>
  <g>
    <path d="M 184,4968 A 8,8 0,0,0 178,4972" class="nofill"></path>
    <line x1="178" y1="4972" x2="176" y2="4976" class="solid"></line>
    <line x1="184" y1="4968" x2="240" y2="4968" class="solid"></line>
    <line x1="240" y1="4968" x2="248" y2="4976" class="solid"></line>
    <line x1="248" y1="4976" x2="320" y2="4976" class="solid"></line>
    <path d="M 176,4976 A 16,16 0,0,0 176,4992" class="nofill"></path>
    <line x1="320" y1="4976" x2="328" y2="4984" class="solid"></line>
    <line x1="176" y1="4992" x2="178" y2="4996" class="solid"></line>
    <path d="M 178,4996 A 8,8 0,0,0 184,5000" class="nofill"></path>
    <line x1="184" y1="5000" x2="216" y2="5000" class="solid"></line>
    <line x1="224" y1="4992" x2="296" y2="4992" class="solid"></line>
    <line x1="224" y1="4992" x2="216" y2="5000" class="solid"></line>
    <line x1="296" y1="4992" x2="304" y2="5000" class="solid"></line>
    <line x1="304" y1="5000" x2="336" y2="5000" class="solid"></line>
  </g>
  <g>
    <line x1="244" y1="5024" x2="296" y2="5024" class="solid"></line>
    <line x1="304" y1="5016" x2="296" y2="5024" class="solid"></line>
    <line x1="304" y1="5016" x2="336" y2="5016" class="solid"></line>
    <line x1="244" y1="5024" x2="244" y2="5124" class="solid"></line>
    <path d="M 216,5128 A 4,4 0,0,0 212,5132" class="nofill"></path>
    <line x1="212" y1="5132" x2="212" y2="5168" class="solid"></line>
    <line x1="216" y1="5128" x2="240" y2="5128" class="solid"></line>
    <path d="M 244,5124 A 4,4 0,0,1 240,5128" class="nofill"></path>
    <line x1="212" y1="5160" x2="240" y2="5160" class="solid"></line>
    <path d="M 240,5160 A 4,4 0,0,1 244,5164" class="nofill"></path>
    <line x1="244" y1="5164" x2="244" y2="5220" class="solid"></line>
    <line x1="136" y1="5200" x2="164" y2="5200" class="solid"></line>
    <line x1="164" y1="5200" x2="164" y2="5220" class="solid"></line>
    <path d="M 164,5220 A 4,4 0,0,0 168,5224" class="nofill"></path>
    <line x1="180" y1="5184" x2="180" y2="5224" class="solid"></line>
    <line x1="196" y1="5184" x2="196" y2="5224" class="solid"></line>
    <line x1="212" y1="5184" x2="212" y2="5224" class="solid"></line>
    <line x1="228" y1="5184" x2="228" y2="5224" class="solid"></line>
    <line x1="168" y1="5224" x2="240" y2="5224" class="solid"></line>
    <path d="M 244,5220 A 4,4 0,0,1 240,5224" class="nofill"></path>
  </g>
  <g>
    <line x1="268" y1="5020" x2="268" y2="5108" class="solid"></line>
    <path d="M 268,5108 A 4,4 0,0,0 272,5112" class="nofill"></path>
    <line x1="272" y1="5112" x2="368" y2="5112" class="solid"></line>
    <path d="M 368,5112 A 4,4 0,0,1 372,5116" class="nofill"></path>
    <line x1="372" y1="5116" x2="372" y2="5172" class="solid"></line>
  </g>
  <g>
    <line x1="540" y1="5144" x2="620" y2="5144" class="solid"></line>
    <line x1="540" y1="5144" x2="540" y2="5172" class="solid"></line>
    <line x1="556" y1="5144" x2="556" y2="5224" class="solid"></line>
    <line x1="572" y1="5144" x2="572" y2="5224" class="solid"></line>
    <line x1="588" y1="5144" x2="588" y2="5224" class="solid"></line>
    <line x1="604" y1="5144" x2="604" y2="5224" class="solid"></line>
    <line x1="620" y1="5144" x2="620" y2="5224" class="solid"></line>
    <line x1="540" y1="5160" x2="620" y2="5160" class="solid"></line>
    <line x1="544" y1="5176" x2="620" y2="5176" class="solid"></line>
    <line x1="544" y1="5192" x2="620" y2="5192" class="solid"></line>
    <line x1="540" y1="5196" x2="540" y2="5224" class="solid"></line>
    <line x1="540" y1="5208" x2="620" y2="5208" class="solid"></line>
    <line x1="540" y1="5224" x2="620" y2="5224" class="solid"></line>
  </g>
  <g>
    <line x1="272" y1="5168" x2="288" y2="5168" class="solid"></line>
    <line x1="288" y1="5168" x2="296" y2="5176" class="solid"></line>
  </g>
  <g>
    <line x1="348" y1="5224" x2="348" y2="5272" class="solid"></line>
    <line x1="424" y1="5224" x2="444" y2="5224" class="solid"></line>
    <line x1="444" y1="5224" x2="444" y2="5272" class="solid"></line>
    <line x1="348" y1="5272" x2="444" y2="5272" class="solid"></line>
  </g>
  <g>
    <path d="M 64,5192 A 3,3 0,0,0 62,5196" class="nofill"></path>
    <line x1="62" y1="5196" x2="72" y2="5216" class="solid"></line>
    <line x1="64" y1="5192" x2="120" y2="5192" class="solid"></line>
  </g>
  <g>
    <path d="M 192,5432 A 8,8 0,0,0 186,5436" class="nofill"></path>
    <line x1="186" y1="5436" x2="184" y2="5440" class="solid"></line>
    <line x1="192" y1="5432" x2="224" y2="5432" class="solid"></line>
    <path d="M 224,5432 A 4,4 0,0,1 228,5436" class="nofill"></path>
    <line x1="228" y1="5436" x2="228" y2="5524" class="solid"></line>
    <path d="M 184,5440 A 16,16 0,0,0 184,5456" class="nofill"></path>
    <path d="M 192,5448 A 8,8 0,0,0 186,5452" class="nofill"></path>
    <line x1="186" y1="5452" x2="184" y2="5456" class="solid"></line>
    <line x1="192" y1="5448" x2="200" y2="5448" class="solid"></line>
    <path d="M 184,5456 A 16,16 0,0,0 184,5472" class="nofill"></path>
    <path d="M 208,5440 A 16,16 0,0,1 208,5456" class="nofill"></path>
    <line x1="208" y1="5456" x2="206" y2="5460" class="solid"></line>
    <path d="M 208,5456 A 16,16 0,0,1 208,5472" class="nofill"></path>
    <line x1="208" y1="5472" x2="206" y2="5476" class="solid"></line>
    <path d="M 208,5472 A 16,16 0,0,1 208,5488" class="nofill"></path>
    <line x1="208" y1="5488" x2="206" y2="5492" class="solid"></line>
    <path d="M 192,5464 A 8,8 0,0,0 186,5468" class="nofill"></path>
    <line x1="186" y1="5468" x2="184" y2="5472" class="solid"></line>
    <line x1="192" y1="5464" x2="200" y2="5464" class="solid"></line>
    <path d="M 206,5460 A 8,8 0,0,1 200,5464" class="nofill"></path>
    <path d="M 184,5472 A 16,16 0,0,0 184,5488" class="nofill"></path>
    <path d="M 192,5480 A 8,8 0,0,0 186,5484" class="nofill"></path>
    <line x1="186" y1="5484" x2="184" y2="5488" class="solid"></line>
    <line x1="192" y1="5480" x2="200" y2="5480" class="solid"></line>
    <path d="M 206,5476 A 8,8 0,0,1 200,5480" class="nofill"></path>
    <path d="M 180,5488 A 12,12 0,0,0 180,5504" class="nofill"></path>
    <path d="M 188,5496 A 8,8 0,0,0 180,5504" class="nofill"></path>
    <line x1="188" y1="5496" x2="200" y2="5496" class="solid"></line>
    <path d="M 206,5492 A 8,8 0,0,1 200,5496" class="nofill"></path>
    <line x1="180" y1="5504" x2="180" y2="5524" class="solid"></line>
    <line x1="260" y1="5504" x2="292" y2="5504" class="solid"></line>
    <line x1="260" y1="5504" x2="260" y2="5524" class="solid"></line>
    <line x1="260" y1="5520" x2="292" y2="5520" class="solid"></line>
    <line x1="292" y1="5504" x2="292" y2="5524" class="solid"></line>
    <path d="M 144,5528 A 4,4 0,0,0 140,5532" class="nofill"></path>
    <line x1="140" y1="5532" x2="140" y2="5540" class="solid"></line>
    <line x1="144" y1="5528" x2="176" y2="5528" class="solid"></line>
    <path d="M 180,5524 A 4,4 0,0,1 176,5528" class="nofill"></path>
    <path d="M 180,5524 A 4,4 0,0,0 184,5528" class="nofill"></path>
    <line x1="184" y1="5528" x2="224" y2="5528" class="solid"></line>
    <path d="M 228,5524 A 4,4 0,0,1 224,5528" class="nofill"></path>
    <path d="M 228,5524 A 4,4 0,0,0 232,5528" class="nofill"></path>
    <line x1="232" y1="5528" x2="256" y2="5528" class="solid"></line>
    <line x1="264" y1="5520" x2="256" y2="5528" class="solid"></line>
    <path d="M 260,5524 A 4,4 0,0,1 256,5528" class="nofill"></path>
    <path d="M 260,5524 A 4,4 0,0,0 264,5528" class="nofill"></path>
    <line x1="264" y1="5528" x2="288" y2="5528" class="solid"></line>
    <path d="M 292,5524 A 4,4 0,0,1 288,5528" class="nofill"></path>
    <path d="M 140,5540 A 4,4 0,0,0 144,5544" class="nofill"></path>
    <line x1="144" y1="5544" x2="296" y2="5544" class="solid"></line>
  </g>
  <g>
    <path d="M 360,5672 A 8,8 0,0,0 354,5676" class="nofill"></path>
    <line x1="360" y1="5672" x2="376" y2="5672" class="solid"></line>
  </g>
  <g>
    <path d="M 344,5704 A 8,8 0,0,0 338,5708" class="nofill"></path>
    <line x1="344" y1="5704" x2="376" y2="5704" class="solid"></line>
  </g>
  <g>
    <line x1="160" y1="5720" x2="216" y2="5720" class="solid"></line>
    <path d="M 216,5720 A 8,8 0,0,1 222,5724" class="nofill"></path>
  </g>
  <g>
    <path d="M 256,5832 A 8,8 0,0,0 250,5836" class="nofill"></path>
    <line x1="250" y1="5836" x2="248" y2="5840" class="solid"></line>
    <line x1="256" y1="5832" x2="280" y2="5832" class="solid"></line>
    <path d="M 280,5832 A 8,8 0,0,1 286,5836" class="nofill"></path>
    <line x1="286" y1="5836" x2="288" y2="5840" class="solid"></line>
    <path d="M 248,5840 A 16,16 0,0,0 248,5856" class="nofill"></path>
    <path d="M 288,5840 A 16,16 0,0,1 288,5856" class="nofill"></path>
    <line x1="248" y1="5856" x2="250" y2="5860" class="solid"></line>
    <path d="M 250,5860 A 8,8 0,0,0 256,5864" class="nofill"></path>
    <line x1="256" y1="5864" x2="280" y2="5864" class="solid"></line>
    <line x1="288" y1="5856" x2="286" y2="5860" class="solid"></line>
    <path d="M 286,5860 A 8,8 0,0,1 280,5864" class="nofill"></path>
  </g>
  <g>
    <line x1="304" y1="5832" x2="320" y2="5832" class="solid"></line>
    <line x1="320" y1="5832" x2="328" y2="5840" class="solid"></line>
    <line x1="328" y1="5840" x2="352" y2="5840" class="solid"></line>
    <line x1="352" y1="5840" x2="360" y2="5848" class="solid"></line>
  </g>
  <g>
    <line x1="304" y1="5872" x2="330" y2="5924" class="solid"></line>
    <path d="M 330,5924 A 8,8 0,0,0 336,5928" class="nofill"></path>
    <line x1="336" y1="5928" x2="384" y2="5928" class="solid"></line>
    <line x1="372" y1="5928" x2="386" y2="5956" class="solid"></line>
    <path d="M 386,5956 A 8,8 0,0,0 392,5960" class="nofill"></path>
    <line x1="392" y1="5960" x2="424" y2="5960" class="solid"></line>
  </g>
  <g>
    <line x1="248" y1="5888" x2="238" y2="5908" class="solid"></line>
    <path d="M 216,5912 A 8,8 0,0,0 210,5916" class="nofill"></path>
    <line x1="210" y1="5916" x2="196" y2="5944" class="solid"></line>
    <line x1="216" y1="5912" x2="232" y2="5912" class="solid"></line>
    <path d="M 238,5908 A 8,8 0,0,1 232,5912" class="nofill"></path>
    <line x1="196" y1="5944" x2="196" y2="5976" class="solid"></line>
    <line x1="196" y1="5976" x2="174" y2="6020" class="solid"></line>
  </g>
  <g>
    <line x1="280" y1="5888" x2="344" y2="6016" class="solid"></line>
    <line x1="344" y1="6016" x2="354" y2="6036" class="solid"></line>
    <path d="M 354,6036 A 8,8 0,0,0 360,6040" class="nofill"></path>
    <line x1="360" y1="6040" x2="376" y2="6040" class="solid"></line>
  </g>
  <g>
    <line x1="296" y1="5888" x2="346" y2="5988" class="solid"></line>
    <path d="M 346,5988 A 8,8 0,0,0 352,5992" class="nofill"></path>
    <line x1="352" y1="5992" x2="392" y2="5992" class="solid"></line>
  </g>
  <g>
    <line x1="196" y1="5984" x2="196" y2="6008" class="solid"></line>
    <line x1="196" y1="6008" x2="174" y2="6052" class="solid"></line>
  </g>
  <g>
    <line x1="196" y1="6016" x2="196" y2="6036" class="solid"></line>
    <path d="M 196,6036 A 16,16 0,0,1 194,6044" class="nofill"></path>
    <line x1="194" y1="6044" x2="174" y2="6084" class="solid"></line>
  </g>
  <g>
    <line x1="408" y1="5712" x2="418" y2="5732" class="solid"></line>
    <path d="M 418,5732 A 8,8 0,0,0 424,5736" class="nofill"></path>
    <line x1="424" y1="5736" x2="448" y2="5736" class="solid"></line>
  </g>
  <g>
    <line x1="424" y1="5744" x2="434" y2="5764" class="solid"></line>
    <path d="M 434,5764 A 8,8 0,0,0 440,5768" class="nofill"></path>
    <line x1="440" y1="5768" x2="456" y2="5768" class="solid"></line>
  </g>
  <g>
    <line x1="136" y1="5776" x2="170" y2="5844" class="solid"></line>
    <path d="M 170,5844 A 8,8 0,0,0 176,5848" class="nofill"></path>
    <line x1="176" y1="5848" x2="224" y2="5848" class="solid"></line>
  </g>
  <g>
    <line x1="464" y1="5824" x2="520" y2="5824" class="solid"></line>
    <line x1="464" y1="5824" x2="456" y2="5840" class="solid"></line>
    <line x1="520" y1="5824" x2="528" y2="5840" class="solid"></line>
  </g>
  <g>
    <line x1="456" y1="5856" x2="464" y2="5872" class="solid"></line>
    <line x1="464" y1="5872" x2="520" y2="5872" class="solid"></line>
    <line x1="528" y1="5856" x2="520" y2="5872" class="solid"></line>
  </g>
  <g>
    <line x1="196" y1="6208" x2="228" y2="6208" class="solid"></line>
    <line x1="184" y1="6216" x2="196" y2="6216" class="solid"></line>
    <line x1="196" y1="6208" x2="196" y2="6224" class="solid"></line>
    <line x1="196" y1="6224" x2="228" y2="6224" class="solid"></line>
    <line x1="228" y1="6208" x2="228" y2="6224" class="solid"></line>
  </g>
  <g>
    <line x1="132" y1="6220" x2="132" y2="6256" class="solid"></line>
    <line x1="120" y1="6256" x2="144" y2="6256" class="solid"></line>
  </g>
  <g>
    <line x1="180" y1="6220" x2="180" y2="6292" class="solid"></line>
    <path d="M 180,6292 A 4,4 0,0,0 184,6296" class="nofill"></path>
    <line x1="184" y1="6296" x2="200" y2="6296" class="solid"></line>
    <path d="M 200,6296 A 4,4 0,0,1 204,6300" class="nofill"></path>
    <line x1="204" y1="6300" x2="204" y2="6328" class="solid"></line>
    <line x1="244" y1="6220" x2="244" y2="6328" class="solid"></line>
    <line x1="268" y1="6220" x2="268" y2="6248" class="solid"></line>
    <path d="M 264,6248 A 4,4 0,0,0 260,6252" class="nofill"></path>
    <line x1="260" y1="6252" x2="260" y2="6292" class="solid"></line>
    <line x1="264" y1="6248" x2="272" y2="6248" class="solid"></line>
    <path d="M 272,6248 A 4,4 0,0,1 276,6252" class="nofill"></path>
    <line x1="276" y1="6252" x2="276" y2="6292" class="solid"></line>
    <path d="M 260,6292 A 4,4 0,0,0 264,6296" class="nofill"></path>
    <line x1="264" y1="6296" x2="272" y2="6296" class="solid"></line>
    <line x1="268" y1="6296" x2="268" y2="6328" class="solid"></line>
    <path d="M 276,6292 A 4,4 0,0,1 272,6296" class="nofill"></path>
    <path d="M 192,6328 A 4,4 0,0,0 188,6332" class="nofill"></path>
    <line x1="188" y1="6332" x2="188" y2="6580" class="solid"></line>
    <line x1="192" y1="6328" x2="288" y2="6328" class="solid"></line>
    <path d="M 288,6328 A 4,4 0,0,1 292,6332" class="nofill"></path>
    <line x1="292" y1="6332" x2="292" y2="6580" class="solid"></line>
    <line x1="292" y1="6568" x2="356" y2="6568" class="solid"></line>
    <line x1="356" y1="6560" x2="356" y2="6576" class="solid"></line>
    <path d="M 188,6580 A 4,4 0,0,0 192,6584" class="nofill"></path>
    <line x1="192" y1="6584" x2="288" y2="6584" class="solid"></line>
    <line x1="236" y1="6584" x2="236" y2="6644" class="solid"></line>
    <line x1="276" y1="6584" x2="276" y2="6612" class="solid"></line>
    <path d="M 292,6580 A 4,4 0,0,1 288,6584" class="nofill"></path>
    <path d="M 276,6612 A 4,4 0,0,0 280,6616" class="nofill"></path>
    <path d="M 128,6424 A 4,4 0,0,0 124,6428" class="nofill"></path>
    <line x1="124" y1="6428" x2="124" y2="6528" class="solid"></line>
    <line x1="128" y1="6424" x2="188" y2="6424" class="solid"></line>
    <line x1="96" y1="6528" x2="168" y2="6528" class="solid"></line>
  </g>
  <g>
    <line x1="348" y1="6220" x2="348" y2="6296" class="solid"></line>
    <path d="M 344,6296 A 4,4 0,0,0 340,6300" class="nofill"></path>
    <line x1="340" y1="6300" x2="340" y2="6340" class="solid"></line>
    <line x1="344" y1="6296" x2="352" y2="6296" class="solid"></line>
    <path d="M 352,6296 A 4,4 0,0,1 356,6300" class="nofill"></path>
    <line x1="356" y1="6300" x2="356" y2="6340" class="solid"></line>
    <path d="M 340,6340 A 4,4 0,0,0 344,6344" class="nofill"></path>
    <line x1="344" y1="6344" x2="352" y2="6344" class="solid"></line>
    <line x1="348" y1="6344" x2="348" y2="6372" class="solid"></line>
    <path d="M 356,6340 A 4,4 0,0,1 352,6344" class="nofill"></path>
  </g>
  <g>
    <line x1="388" y1="6220" x2="388" y2="6344" class="solid"></line>
    <line x1="388" y1="6344" x2="372" y2="6376" class="solid"></line>
    <line x1="372" y1="6352" x2="372" y2="6400" class="solid"></line>
    <line x1="352" y1="6376" x2="372" y2="6376" class="solid"></line>
  </g>
  <g>
    <line x1="392" y1="6216" x2="448" y2="6216" class="solid"></line>
    <path d="M 448,6216 A 4,4 0,0,1 452,6220" class="nofill"></path>
    <line x1="452" y1="6220" x2="452" y2="6288" class="solid"></line>
    <line x1="440" y1="6288" x2="464" y2="6288" class="solid"></line>
    <line x1="440" y1="6288" x2="452" y2="6312" class="solid"></line>
    <line x1="464" y1="6288" x2="452" y2="6312" class="solid"></line>
    <line x1="440" y1="6312" x2="464" y2="6312" class="solid"></line>
    <line x1="452" y1="6312" x2="452" y2="6424" class="solid"></line>
    <line x1="436" y1="6416" x2="436" y2="6464" class="solid"></line>
    <line x1="436" y1="6424" x2="452" y2="6424" class="solid"></line>
    <line x1="436" y1="6440" x2="448" y2="6440" class="solid"></line>
    <line x1="436" y1="6456" x2="452" y2="6456" class="solid"></line>
  </g>
  <g>
    <line x1="128" y1="6264" x2="140" y2="6264" class="solid"></line>
    <path d="M 140,6264 A 8,8 0,0,1 148,6272" class="nofill"></path>
  </g>
  <g>
    <line x1="132" y1="6316" x2="132" y2="6336" class="solid"></line>
    <line x1="104" y1="6336" x2="160" y2="6336" class="solid"></line>
  </g>
  <g>
    <line x1="348" y1="6380" x2="348" y2="6408" class="solid"></line>
    <path d="M 344,6408 A 4,4 0,0,0 340,6412" class="nofill"></path>
    <line x1="340" y1="6412" x2="340" y2="6452" class="solid"></line>
    <line x1="344" y1="6408" x2="352" y2="6408" class="solid"></line>
    <path d="M 352,6408 A 4,4 0,0,1 356,6412" class="nofill"></path>
    <line x1="356" y1="6412" x2="356" y2="6452" class="solid"></line>
    <path d="M 340,6452 A 4,4 0,0,0 344,6456" class="nofill"></path>
    <line x1="344" y1="6456" x2="352" y2="6456" class="solid"></line>
    <line x1="348" y1="6456" x2="348" y2="6484" class="solid"></line>
    <path d="M 356,6452 A 4,4 0,0,1 352,6456" class="nofill"></path>
    <path d="M 348,6484 A 4,4 0,0,0 352,6488" class="nofill"></path>
    <line x1="352" y1="6488" x2="384" y2="6488" class="solid"></line>
    <line x1="388" y1="6428" x2="388" y2="6484" class="solid"></line>
    <path d="M 388,6484 A 4,4 0,0,1 384,6488" class="nofill"></path>
  </g>
  <g>
    <line x1="392" y1="6424" x2="428" y2="6424" class="solid"></line>
    <line x1="428" y1="6416" x2="428" y2="6464" class="solid"></line>
  </g>
  <g>
    <line x1="456" y1="6488" x2="476" y2="6488" class="solid"></line>
    <line x1="476" y1="6480" x2="476" y2="6496" class="solid"></line>
  </g>
  <g>
    <line x1="124" y1="6544" x2="124" y2="6644" class="solid"></line>
    <path d="M 124,6644 A 4,4 0,0,0 128,6648" class="nofill"></path>
  </g>
  <g>
    <line x1="364" y1="6560" x2="364" y2="6576" class="solid"></line>
    <line x1="364" y1="6568" x2="384" y2="6568" class="solid"></line>
    <path d="M 384,6568 A 4,4 0,0,1 388,6572" class="nofill"></path>
    <line x1="388" y1="6572" x2="388" y2="6612" class="solid"></line>
  </g>
  <g>
    <line x1="500" y1="6620" x2="500" y2="6644" class="solid"></line>
    <line x1="452" y1="6640" x2="484" y2="6640" class="solid"></line>
    <line x1="432" y1="6648" x2="452" y2="6648" class="solid"></line>
    <line x1="452" y1="6640" x2="452" y2="6656" class="solid"></line>
    <line x1="452" y1="6656" x2="484" y2="6656" class="solid"></line>
    <line x1="484" y1="6640" x2="484" y2="6656" class="solid"></line>
    <line x1="484" y1="6648" x2="496" y2="6648" class="solid"></line>
    <path d="M 500,6644 A 4,4 0,0,1 496,6648" class="nofill"></path>
  </g>
  <g>
    <line x1="232" y1="6648" x2="292" y2="6648" class="solid"></line>
    <line x1="296" y1="6640" x2="288" y2="6656" class="solid"></line>
    <line x1="296" y1="6640" x2="304" y2="6656" class="solid"></line>
    <line x1="312" y1="6640" x2="304" y2="6656" class="solid"></line>
    <line x1="312" y1="6640" x2="320" y2="6656" class="solid"></line>
    <line x1="328" y1="6640" x2="320" y2="6656" class="solid"></line>
  </g>
  <g>
    <line x1="428" y1="6652" x2="428" y2="6680" class="solid"></line>
    <path d="M 424,6680 A 4,4 0,0,0 420,6684" class="nofill"></path>
    <line x1="420" y1="6684" x2="420" y2="6724" class="solid"></line>
    <line x1="424" y1="6680" x2="432" y2="6680" class="solid"></line>
    <path d="M 432,6680 A 4,4 0,0,1 436,6684" class="nofill"></path>
    <line x1="436" y1="6684" x2="436" y2="6724" class="solid"></line>
    <path d="M 420,6724 A 4,4 0,0,0 424,6728" class="nofill"></path>
    <line x1="424" y1="6728" x2="432" y2="6728" class="solid"></line>
    <line x1="428" y1="6728" x2="428" y2="6752" class="solid"></line>
    <path d="M 436,6724 A 4,4 0,0,1 432,6728" class="nofill"></path>
  </g>
  <g>
    <line x1="80" y1="6944" x2="136" y2="6944" class="solid"></line>
    <line x1="80" y1="6944" x2="64" y2="6976" class="solid"></line>
    <line x1="136" y1="6944" x2="152" y2="6976" class="solid"></line>
    <line x1="64" y1="6976" x2="80" y2="7008" class="solid"></line>
    <line x1="152" y1="6976" x2="136" y2="7008" class="solid"></line>
    <line x1="80" y1="7008" x2="136" y2="7008" class="solid"></line>
  </g>
  <g>
    <path d="M 256,6888 A 8,8 0,0,0 250,6892" class="nofill"></path>
    <line x1="250" y1="6892" x2="248" y2="6896" class="solid"></line>
    <line x1="256" y1="6888" x2="304" y2="6888" class="solid"></line>
    <path d="M 304,6888 A 8,8 0,0,1 310,6892" class="nofill"></path>
    <line x1="310" y1="6892" x2="312" y2="6896" class="solid"></line>
    <path d="M 248,6896 A 16,16 0,0,0 248,6912" class="nofill"></path>
    <path d="M 312,6896 A 16,16 0,0,1 312,6912" class="nofill"></path>
    <line x1="248" y1="6912" x2="250" y2="6916" class="solid"></line>
    <path d="M 250,6916 A 8,8 0,0,0 256,6920" class="nofill"></path>
    <line x1="256" y1="6920" x2="304" y2="6920" class="solid"></line>
    <line x1="312" y1="6912" x2="310" y2="6916" class="solid"></line>
    <path d="M 310,6916 A 8,8 0,0,1 304,6920" class="nofill"></path>
  </g>
  <g>
    <line x1="232" y1="6944" x2="296" y2="6944" class="solid"></line>
    <line x1="232" y1="6944" x2="240" y2="6960" class="solid"></line>
    <line x1="296" y1="6944" x2="304" y2="6960" class="solid"></line>
    <path d="M 240,6960 A 16,16 0,0,1 240,6976" class="nofill"></path>
    <path d="M 304,6960 A 16,16 0,0,1 304,6976" class="nofill"></path>
    <line x1="240" y1="6976" x2="232" y2="6992" class="solid"></line>
    <line x1="232" y1="6992" x2="296" y2="6992" class="solid"></line>
    <line x1="304" y1="6976" x2="296" y2="6992" class="solid"></line>
  </g>
  <g>
    <line x1="200" y1="6944" x2="184" y2="6976" class="solid"></line>
    <line x1="200" y1="6944" x2="216" y2="6976" class="solid"></line>
    <line x1="184" y1="6976" x2="200" y2="7008" class="solid"></line>
    <line x1="216" y1="6976" x2="200" y2="7008" class="solid"></line>
  </g>
  <g>
    <path d="M 216,7208 A 3,3 0,0,0 214,7212" class="nofill"></path>
    <line x1="214" y1="7212" x2="236" y2="7256" class="solid"></line>
    <line x1="216" y1="7208" x2="256" y2="7208" class="solid"></line>
    <path d="M 256,7208 A 3,3 0,0,1 258,7212" class="nofill"></path>
    <line x1="258" y1="7212" x2="236" y2="7256" class="solid"></line>
  </g>
  <g>
    <path d="M 280,7208 A 3,3 0,0,0 278,7212" class="nofill"></path>
    <line x1="278" y1="7212" x2="298" y2="7252" class="solid"></line>
    <line x1="280" y1="7208" x2="312" y2="7208" class="solid"></line>
    <path d="M 312,7208 A 8,8 0,0,1 318,7212" class="nofill"></path>
    <line x1="318" y1="7212" x2="338" y2="7252" class="solid"></line>
    <path d="M 298,7252 A 8,8 0,0,0 304,7256" class="nofill"></path>
    <line x1="304" y1="7256" x2="336" y2="7256" class="solid"></line>
    <path d="M 338,7252 A 3,3 0,0,1 336,7256" class="nofill"></path>
  </g>
  <g>
    <path d="M 24,7368 A 8,8 0,0,0 18,7372" class="nofill"></path>
    <line x1="18" y1="7372" x2="6" y2="7396" class="solid"></line>
    <line x1="24" y1="7368" x2="128" y2="7368" class="solid"></line>
    <path d="M 128,7368 A 3,3 0,0,1 130,7372" class="nofill"></path>
    <line x1="130" y1="7372" x2="118" y2="7396" class="solid"></line>
    <path d="M 6,7396 A 3,3 0,0,0 8,7400" class="nofill"></path>
    <line x1="8" y1="7400" x2="112" y2="7400" class="solid"></line>
    <path d="M 118,7396 A 8,8 0,0,1 112,7400" class="nofill"></path>
  </g>
</svg>

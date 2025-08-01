loading_html = """
<div style="display: inline-block; margin-left: 10px;">
  <span class="dot-flashing"></span>
</div>
<style>
@keyframes dotFlashing {
  0% { background-color: #2e8be8; }
  80%, 100% { background-color: rgba(46,139,232,0.2); }
}
.dot-flashing {
  width: 10px;
  height: 10px;
  border-radius: 5px;
  background-color: #2e8be8;
  color: #2e8be8;
  animation: dotFlashing 0.5s infinite linear alternate;
  display: inline-block;
}
</style>
"""
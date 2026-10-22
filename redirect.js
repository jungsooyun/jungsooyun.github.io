// 옛 Jekyll 블로그의 주소를 jerrylog.dev로 보낸다. 매핑에 없으면 홈으로 간다.
(function () {
  var base = "https://jerrylog.dev";
  var map = {
    "/mab-survey": "/ko/blog/multi-armed-bandit-history",
    "/mab-survey-eng": "/ko/blog/multi-armed-bandit-history",
  };
  var key = location.pathname.replace(/\/+$/, "").toLowerCase();
  location.replace(base + (map[key] || "/ko"));
})();

// Google Analytics 4, cookieless: Consent Mode keeps analytics and ad storage
// denied, so GA sends anonymous pings and sets no cookies (the page promises no
// tracking cookies, so it needs no consent banner). Its own property; both
// prastara.am2.ai and am2strata.am2.ai report into it (split by hostname).
window.dataLayer = window.dataLayer || [];
function gtag() { dataLayer.push(arguments); }
(() => {
const GA_ID = "G-LT0WDVQYHK"; // GA4 property "Prastara", web stream "Prastara site"
gtag("consent", "default", { analytics_storage: "denied", ad_storage: "denied", ad_user_data: "denied", ad_personalization: "denied" });
gtag("js", new Date());
gtag("config", GA_ID);
const s = document.createElement("script");
s.async = true;
s.src = "https://www.googletagmanager.com/gtag/js?id=" + GA_ID;
document.head.appendChild(s);

// The few clicks that say whether the page works: installs, GitHub, videos, sponsors.
document.addEventListener("click", e => {
  const el = e.target.closest("a, button.copy");
  if (!el) return;
  const href = el.getAttribute("href") || "";
  if (el.matches("button.copy")) gtag("event", "install_copy", { os: el.closest("#cmd-win") ? "windows" : "unix" });
  else if (href.includes("github.com/sponsors")) gtag("event", "sponsor_click");
  else if (href.includes("github.com/abhaymin/")) gtag("event", "github_click", { where: el.closest("nav, .hero, footer")?.localName || "page" });
  else if (/youtube\.com|youtu\.be/.test(href)) gtag("event", "video_open", { title: (el.querySelector("b")?.textContent || href).slice(0, 100) });
});
})();

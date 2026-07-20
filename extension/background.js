// Service worker: adds the right-click menu, cleans the selection into a ticker,
// runs the in-browser analysis, and hands the result to the in-page card.
//
// Lite edition: everything runs client-side in analyzer.js — no server, no native
// host, no setup. The fetches run here (CORS-exempt via host_permissions), not in
// the content script. For the full analysis (peers, full statements) point users to
// the web app.

import { analyzeTicker } from "./analyzer.js";

const MENU_ID = "mra-analyze";

chrome.runtime.onInstalled.addListener(() => {
  chrome.contextMenus.create({
    id: MENU_ID,
    title: 'Analyze "%s" risk',
    contexts: ["selection"],
  });
});

// Pull a plausible ticker out of a messy selection:
//  "$AAPL" -> AAPL,  "Apple (AAPL)" -> AAPL,  "aapl shares" -> AAPL
function cleanTicker(text) {
  if (!text) return "";
  const upper = text.trim().toUpperCase();
  const paren = upper.match(/\(([A-Z]{1,5})\)/);
  if (paren) return paren[1];
  const token = upper.replace(/^[^A-Z$]+/, "").split(/\s+/)[0] || "";
  return token.replace(/[^A-Z.]/g, "").slice(0, 6);
}

async function injectCard(tabId) {
  // activeTab (granted by the context-menu gesture) lets us inject here without
  // broad host permissions. content.js guards against double-initialization.
  try {
    await chrome.scripting.executeScript({ target: { tabId }, files: ["content.js"] });
  } catch (e) {
    // e.g. chrome:// pages or the web store, where injection is disallowed.
    return false;
  }
  return true;
}

function send(tabId, msg) {
  chrome.tabs.sendMessage(tabId, msg).catch(() => {});
}

chrome.contextMenus.onClicked.addListener(async (info, tab) => {
  if (info.menuItemId !== MENU_ID || !tab || !tab.id) return;

  const ticker = cleanTicker(info.selectionText);
  const injected = await injectCard(tab.id);
  if (!injected) return;

  if (!ticker) {
    send(tab.id, { type: "error", message: "Couldn't read a ticker from that selection." });
    return;
  }

  send(tab.id, { type: "loading", ticker });

  try {
    const data = await analyzeTicker(ticker);
    send(tab.id, { type: "render", data });
  } catch (e) {
    send(tab.id, { type: "error", message: "Couldn't analyze " + ticker + " (" + e.message + ")." });
  }
});

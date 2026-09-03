// Cloudflare Worker — Telegram webhook relay.
//
// WHY THIS EXISTS
//   Inbound connections from Telegram's ranges (91.108.4.0/22, 149.154.160.0/20)
//   to our host are dropped upstream — verified on both TCP 443 and 8443, with
//   every local counter clean and outbound flawless. Webhook delivery stalled
//   60-543s. The filter matches on Telegram's SOURCE addresses, so changing our
//   port or hostname does nothing; the only fix is for the connection to arrive
//   from somewhere else.
//
//   This Worker is that somewhere else. Telegram connects to Cloudflare, which
//   is not filtered, and Cloudflare connects to us, which is also not filtered
//   because it is not Telegram. Nothing about our server changes.
//
// DEPLOY
//   Cloudflare dashboard -> Workers & Pages -> Create -> paste this -> Deploy.
//   No token, no tunnel, no domain. The route is <name>.<account>.workers.dev.
//
// SECURITY
//   This does not weaken the bot's authentication. The secret token still
//   travels in the header and is still verified with a fixed-time compare on
//   our side, and the allowlist still gates every command. What this DOES do is
//   let anything on Cloudflare's network reach the origin, so ORIGIN is set to
//   the secret path and the origin's IP allowlist must be widened from
//   Telegram's ranges to Cloudflare's.

const ORIGIN = "https://infra-bot.duckdns.org/tg/CRRp7AOxcmjUJNZXXkR9br5z";

export default {
  async fetch(request) {
    // Only POST is a webhook delivery. Everything else is a scanner that found
    // the workers.dev name; give it nothing to work with.
    if (request.method !== "POST") {
      return new Response("Not found", { status: 404 });
    }

    // Forward the body and the ONE header that matters. The secret token is
    // what the bot authenticates on, so it must survive the hop unchanged —
    // dropping it would make every delivery 401 and look like a bot fault.
    const secret = request.headers.get("X-Telegram-Bot-Api-Secret-Token");

    let upstream;
    try {
      upstream = await fetch(ORIGIN, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          ...(secret ? { "X-Telegram-Bot-Api-Secret-Token": secret } : {}),
        },
        body: request.body,
      });
    } catch (err) {
      // Tell Telegram we failed so it RETRIES. Returning 200 here would make a
      // transient origin outage look like successful delivery and silently drop
      // the command.
      return new Response("origin unreachable: " + err, { status: 502 });
    }

    // Pass the origin's status straight through, for the same reason: the bot
    // answers 200 on anything it accepted, and Telegram's retry logic should
    // see the real answer rather than ours.
    return new Response(null, { status: upstream.status });
  },
};

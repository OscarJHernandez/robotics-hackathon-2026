import { useEffect, useState } from "react";

// A touch screen whose short side is phone-sized, in portrait or landscape. ?phone=1 or ?phone=0 overrides it.
const QUERY = "(pointer: coarse) and (max-width: 940px) and (max-height: 940px)";

function check(): boolean {
  const force = new URLSearchParams(window.location.search).get("phone");
  if (force === "1" || force === "0") return force === "1";
  return window.matchMedia(QUERY).matches && Math.min(window.innerWidth, window.innerHeight) <= 540;
}

export function useIsPhone(): boolean {
  const [phone, setPhone] = useState(check);
  useEffect(() => {
    const update = () => setPhone(check());
    const mql = window.matchMedia(QUERY);
    mql.addEventListener("change", update);
    window.addEventListener("resize", update);
    return () => { mql.removeEventListener("change", update); window.removeEventListener("resize", update); };
  }, []);
  return phone;
}

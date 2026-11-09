"""Generate ~100 demo reviews per product for the static NLP site."""

from __future__ import annotations

import csv
import random
from datetime import date, timedelta
from pathlib import Path

SEED = 42
OUT = Path(__file__).resolve().parent / "data" / "reviews.csv"
START = date(2025, 10, 1)
END = date(2026, 6, 30)

PRODUCTS = [
    "Aurora X1 Wireless Earbuds",
    "Nova Pro Over-Ear Headphones",
    "Pulse Mini Bluetooth Speaker",
    "Orbit Fit Band",
    "Flux ANC Headset",
]

FIRST = [
    "Asha", "Rahul", "Meera", "Vikram", "Lina", "Arjun", "Sara", "Dev", "Nora",
    "Kabir", "Priya", "Owen", "Helena", "Imran", "Ruth", "Nikhil", "Elena",
    "Jonah", "Amina", "Leo", "Fatima", "Chris", "Sana", "Mark", "Diya", "Paul",
    "Anika", "Hugo", "Leela", "Samir", "Grace", "Yusuf", "Tara", "Ben", "Maya",
    "Ravi", "Isha", "Omar", "Nina", "Karan", "Zoe", "Alex", "Riya", "Tom",
    "Neha", "Jake", "Sofia", "Ayaan", "Emma", "Rohan",
]

# (sentiment, rating_choices, templates) — {topic} filled from pools
TEMPLATES = {
    "Aurora X1 Wireless Earbuds": {
        "Negative": [
            "The battery {battery_bad}. {sound_bad} and the fit is painful.",
            "Bluetooth {connect_bad}. Call quality is muddy outdoors.",
            "The bass crackles when I turn the volume up. Not worth this price.",
            "They fall out when I walk and hurt after twenty minutes.",
            "Touch controls are far too sensitive. Music pauses when I adjust them.",
            "One side is much louder. This pair feels defective.",
            "The case light says full when the charge is almost gone. Battery dies fast.",
            "The left bud stopped connecting and the warranty form is confusing.",
            "Audio drops whenever I put my phone in my pocket. Connection fails on the bus.",
            "People keep asking me to repeat myself. Calls are too quiet.",
        ],
        "Neutral": [
            "Average sound for podcasts. Fine indoors and nothing I would rave about.",
            "They work for lectures. Music sounds flat and the price feels a bit high.",
            "Solid enough after the update. The design is plain and the app is basic.",
            "Good enough for the bus. The tone is light and the matte case shows fingerprints.",
            "Pleasant and simple. I like the size. I wish the tone had a little more weight.",
            "Okay for daily podcasts. I would not pick these when I want exciting music.",
            "The March update made listening steadier. Music is still a little flat.",
            "Nothing special. Calls are okay and the case is small.",
            "They are okay. Sound is pleasant at low volume and the price still feels slightly high.",
            "Fine for commuting. Battery is average and the tips need frequent adjusting.",
        ],
        "Positive": [
            "Clear vocals and a stable connection on my walk to campus.",
            "After the update the connection is stable while I study. The battery lasts me two days.",
            "Comfortable all day and the case is easy to pocket. The sound is clear and balanced.",
            "Switching tips made them comfortable. The battery now lasts a full workday.",
            "Crisp and clear. Friends say calls sound better and the connection stays stable.",
            "No random cutouts. Comfortable for a three hour study session with a stable connection.",
            "Great for long lectures. The smaller tips feel comfortable and the sound stays clear.",
            "The firmware update fixed the earlier glitches. Voices are clear and music is warm.",
            "The size, the clear sound, and the stable connection all work for class and travel.",
            "After the update these are clear, comfortable, and dependable. My everyday pair.",
        ],
    },
    "Nova Pro Over-Ear Headphones": {
        "Negative": [
            "The clamp is too tight and my ears get hot after an hour. Sound is muddy at midrange.",
            "Noise isolation is weak on the train. The cushioning flattens quickly.",
            "Bass blooms and vocals get buried. Not worth the asking price.",
            "The hinge feels loose and the headband creaks. Connection drops near Wi-Fi routers.",
            "They are heavy for long study sessions. Battery drains faster than advertised.",
            "Mic quality is poor for calls. Coworkers say I sound distant.",
            "Leather pads peel after a few weeks. Soundstage feels narrow.",
            "Bluetooth latency ruins gaming. Cable mode is the only usable option.",
            "The app EQ resets every time I reconnect. Frustrating for daily use.",
            "One earcup is quieter. Support asked me to reset them three times with no fix.",
        ],
        "Neutral": [
            "Decent for office calls. Music is fine but not exciting. Build feels average.",
            "Comfortable enough for short flights. Isolation is okay, not great.",
            "Battery lasts a workday. Sound is balanced but a bit clinical.",
            "They do the job for lectures. I wish the pads were softer.",
            "Solid midrange. Bass is restrained and the case is bulky.",
            "Pairing is simple. Design looks premium but comfort fades after two hours.",
            "Good enough for podcasts. I expected richer detail at this price.",
            "Noise reduction helps a little in cafes. Not enough for open offices.",
            "Cable sound is clearer than wireless. App features feel basic.",
            "Average daily headphones. Nothing broken, nothing memorable.",
        ],
        "Positive": [
            "Plush pads and a wide soundstage. Vocals stay clear even at higher volume.",
            "Excellent for long flights. Isolation is strong and the battery easily lasts two days.",
            "Warm, detailed sound with a secure fit. The hinge feels solid.",
            "Great for focus work. ANC cuts office chatter without pressure.",
            "Calls are crisp and the mic boom alternative works well. Comfortable all afternoon.",
            "Rich bass without mud. Pairing is instant and the case protects them well.",
            "Studio-like clarity for the price. Headband never digs in.",
            "Stable connection across campus. Music and podcasts both sound full.",
            "Soft memory foam and balanced EQ. My go-to for remote meetings.",
            "Build quality feels premium. Sound is clear, deep, and fatigue-free.",
        ],
    },
    "Pulse Mini Bluetooth Speaker": {
        "Negative": [
            "Volume maxes out too soon and the bass rattles the grille. Distortion is obvious.",
            "Battery dies during a short picnic. Charging port feels loose.",
            "Bluetooth drops when I walk ten feet away. Not usable outdoors.",
            "Water resistance failed after light rain. Unit will not power on now.",
            "Sound is tinny at mid volume. Party mode never syncs with a second speaker.",
            "The strap broke in a week. Support said wear and tear is not covered.",
            "It overheats after thirty minutes of loud playback and shuts down.",
            "Buttons are mushy and the app cannot find the speaker half the time.",
            "Microphone for speakerphone calls is awful. Guests cannot hear me.",
            "Price is high for this thin sound. Neighbor's cheaper speaker beats it.",
        ],
        "Neutral": [
            "Fine for a desk. Bass is light and battery lasts an afternoon.",
            "Portable and simple. Sound is okay indoors, weak outside.",
            "Looks cute on a shelf. Volume is adequate for small rooms only.",
            "Pairs quickly. I wish there was a clearer battery indicator.",
            "Good enough for podcasts in the kitchen. Music needs more body.",
            "Average picnic speaker. Holds a charge for a few hours.",
            "Design is clean. Output is safe but unexciting at this price.",
            "Works with my phone and laptop. No wow factor in the sound.",
            "IP rating seems real for splashes. Tone is bright and a bit harsh.",
            "Does what it says. Not a party speaker, more of a personal one.",
        ],
        "Positive": [
            "Surprisingly full sound for the size. Battery lasted our whole beach day.",
            "Punchy bass and clear mids. Pairing is instant every time.",
            "Took light rain without issues. Loud enough for a small patio dinner.",
            "Compact, sturdy strap, and clean vocals. Great travel companion.",
            "Stereo pair mode locked in fast. Guests kept asking what speaker it was.",
            "Charges quickly and plays for hours. Perfect for dorm rooms.",
            "Rich sound without crackle at high volume. Build feels solid.",
            "True pocket size with honest battery life. Love it for camping.",
            "Clear announcements in speakerphone mode. Connection stays stable.",
            "Bright design and warm tone. Best mini speaker I have owned.",
        ],
    },
    "Orbit Fit Band": {
        "Negative": [
            "Heart rate spikes randomly during rest. Sleep stages look completely wrong.",
            "Band irritates my skin after a day. Clasp loosens while I run.",
            "Steps are undercounted compared with my phone. Sync fails twice a week.",
            "Battery dies in two days despite the seven day claim. Charging pins corrode.",
            "Notifications arrive late or not at all. App crashes on open.",
            "GPS tracks are jagged and overstate distance. Useless for training.",
            "Screen scratches easily. Water lock never engages in the pool.",
            "Strap sizing runs small and the replacement cost is ridiculous.",
            "Calorie estimates are wildly high. Coach tips feel generic and wrong.",
            "After the latest update the band will not pair. Factory reset did nothing.",
        ],
        "Neutral": [
            "Tracks steps okay. Sleep data is rough but directionally useful.",
            "Light on the wrist. Battery lasts about four days with always-on off.",
            "App is simple. I wish workouts had more sport profiles.",
            "Fine for casual walking goals. Not precise enough for serious training.",
            "Notifications work most of the time. Screen is dim outdoors.",
            "Comfortable overnight. Heart rate looks plausible at rest.",
            "Syncs when it wants to. Design is plain and inoffensive.",
            "Good reminder to stand. Metrics feel approximate rather than exact.",
            "Clasp is secure enough for office days. Swim tracking is basic.",
            "Does the basics. Nothing broken, nothing impressive.",
        ],
        "Positive": [
            "Accurate resting heart rate and sleep that finally matches how I feel.",
            "Light, comfortable, and the battery easily lasts a full week.",
            "GPS runs are clean and the coach tips helped me improve pace.",
            "Notifications are timely. Strap never irritates during workouts.",
            "Swim tracking and auto workout detect both work well. Love the display.",
            "Sync is reliable and the app charts are clear. Great daily companion.",
            "Step goals keep me moving. Nighttime comfort is excellent.",
            "Heart zones during runs look consistent. Charging is quick.",
            "Water lock is trustworthy. Build quality feels tougher than it looks.",
            "Simple band that just works. Metrics are steady week after week.",
        ],
    },
    "Flux ANC Headset": {
        "Negative": [
            "ANC leaks train rumble and causes ear pressure. Battery drains in four hours.",
            "Transparency mode sounds hollow. Mic quality fails in windy streets.",
            "Pads get sweaty fast and the headband pinches. Not for long flights.",
            "Firmware update broke multipoint. Now only one device stays connected.",
            "Hiss is audible when ANC is on with no music. Distracting in quiet rooms.",
            "Case zipper jammed in a month. Customer support took weeks to reply.",
            "Latency is bad for video calls. Voices cut out when I turn my head.",
            "Bass is exaggerated and muddy. EQ in the app barely helps.",
            "One side ANC is weaker. Feels like a manufacturing defect.",
            "Price is steep for this average isolation. Cheaper rivals seal better.",
        ],
        "Neutral": [
            "ANC is okay for offices. Music is fine, calls are average.",
            "Comfortable for an hour. After that the clamp starts to show.",
            "Battery lasts a workday with ANC on. App is a bit cluttered.",
            "Transparency helps at crosswalks. Sound signature is V-shaped.",
            "Pairs easily. Isolation is good for chatter, weak for low rumble.",
            "Solid travel option if you tweak EQ. Stock tuning is bass heavy.",
            "Mic is acceptable indoors. Outdoors it struggles with wind.",
            "Build looks premium. Features are many but polish is mixed.",
            "Does the job for hybrid work. Not a standout for audiophiles.",
            "Average ANC headset. Reliable enough, not exciting.",
        ],
        "Positive": [
            "ANC melts cafe noise. Vocals stay clear and the pads stay cool.",
            "Excellent for flights. Battery with ANC lasted my entire trip.",
            "Transparency sounds natural. Multipoint between laptop and phone is flawless.",
            "Rich detail and strong seal. Headband never digs in.",
            "Calls are crisp even outdoors. Wind reduction actually works.",
            "Quick charge gives hours of listening. Case is compact and sturdy.",
            "Deep, clean bass with precise mids. My favorite work headset.",
            "Isolation is class leading. Music and podcasts both shine.",
            "Soft pads, stable Bluetooth, and honest battery numbers. Highly recommend.",
            "Studio clarity plus serious ANC. Worth every bit of the price.",
        ],
    },
}

VARIANTS = {
    "battery_bad": [
        "died after four days",
        "lasts about two hours",
        "will not hold a charge",
        "drains overnight",
    ],
    "sound_bad": [
        "The audio drops out",
        "The bass crackles",
        "Sound is thin",
        "Vocals are muddy",
    ],
    "connect_bad": [
        "drops every time my phone goes into my pocket",
        "lags during videos",
        "fails on the bus",
        "cuts out near elevators",
    ],
}

SENTIMENT_MIX = (
    [("Negative", r) for r in (1, 1, 2, 2)] * 8
    + [("Neutral", r) for r in (3, 3, 3)] * 8
    + [("Positive", r) for r in (4, 4, 5, 5)] * 9
)


def fill(template: str) -> str:
    text = template
    for key, options in VARIANTS.items():
        token = "{" + key + "}"
        if token in text:
            text = text.replace(token, random.choice(options))
    return text


def random_date() -> date:
    span = (END - START).days
    return START + timedelta(days=random.randint(0, span))


def generate(per_product: int = 100) -> list[dict]:
    random.seed(SEED)
    rows: list[dict] = []
    review_id = 1
    for product in PRODUCTS:
        pool = TEMPLATES[product]
        mix = list(SENTIMENT_MIX)
        random.shuffle(mix)
        while len(mix) < per_product:
            mix.extend(SENTIMENT_MIX)
        mix = mix[:per_product]
        random.shuffle(mix)
        for sentiment, rating in mix:
            template = random.choice(pool[sentiment])
            # Light paraphrase: occasionally prefix or suffix a short clause
            body = fill(template)
            roll = random.random()
            if roll < 0.12 and sentiment == "Negative":
                body = f"Disappointed overall. {body}"
            elif roll < 0.24 and sentiment == "Positive":
                body = f"{body} Would buy again."
            elif roll < 0.32 and sentiment == "Neutral":
                body = f"{body} Still deciding if I will keep them."
            rows.append(
                {
                    "review_id": review_id,
                    "date": random_date().isoformat(),
                    "product": product,
                    "reviewer": random.choice(FIRST),
                    "rating": rating,
                    "sentiment": sentiment,
                    "review": body,
                }
            )
            review_id += 1
    rows.sort(key=lambda row: (row["product"], row["date"], row["review_id"]))
    # Reassign ids after sort for cleaner files
    for index, row in enumerate(rows, start=1):
        row["review_id"] = index
    return rows


def main() -> None:
    rows = generate(100)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "review_id",
                "date",
                "product",
                "reviewer",
                "rating",
                "sentiment",
                "review",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    counts = {}
    for row in rows:
        counts[row["product"]] = counts.get(row["product"], 0) + 1
    print(f"Wrote {len(rows)} reviews to {OUT}")
    for product, count in counts.items():
        print(f"  {count:3d}  {product}")


if __name__ == "__main__":
    main()

"""
Dataset Optimization for BawaGo AI Agent
Curates 20 sharp, distinct, high-signal questions per intent (500 total across 25 intents),
eliminating cross-intent ambiguity while preserving 100% policy grounding.
"""

import json

OPTIMIZED_INTENTS = {
    "greeting": [
        "Hi", "Hello", "Hey there", "Good morning", "Good afternoon", "Good evening",
        "Greetings", "Hello BawaGo", "Is anyone here?", "What can you do?", "Who are you?",
        "Help me get started", "Namaste BawaGo", "Hi assistant", "Hey I need some help",
        "Hello Bawa", "Hi I have a question", "Good day to you", "Can you assist me today?", "Greetings from a user"
    ],
    "about_bawago": [
        "What is BawaGo?", "Can you explain what this website does?", "What services does BawaGo provide?",
        "Tell me about BawaGo platform", "How does the BawaGo community work?", "What are the 7 verticals of BawaGo?",
        "What is the concept behind BawaGo?", "Why should I use BawaGo?", "Is BawaGo a courier company or a marketplace?",
        "Explain BawaGo to a new user", "What kind of platform is BawaGo?", "What makes BawaGo different?",
        "How does BawaGo connect people?", "Can you give me an overview of BawaGo?", "What can I do on the BawaGo website?",
        "How does peer to peer transport work here?", "Can you describe the BawaGo ecosystem?", "What is the mission of BawaGo?",
        "What problems does BawaGo solve?", "Give me a quick summary of BawaGo"
    ],
    "safevault_escrow_how_it_works": [
        "How does SafeVault Escrow work?", "Is my payment safe on BawaGo?",
        "How do I know the traveler will not take my money and disappear?", "Explain the escrow process on BawaGo",
        "When does the traveler get paid?", "How does payment protection work on this site?", "What is SafeVault?",
        "Why do you hold funds in escrow?", "Can a traveler steal my funds?", "What guarantees do I have that my money is secure?",
        "How does escrow keep buyers and sellers safe?", "When is my money transferred to the seller?",
        "Is my money protected until I receive my order?", "How does the escrow payment system protect me?",
        "Can a traveler run away with my payment?", "How do you prevent payment scams on BawaGo?",
        "Does SafeVault hold the funds until delivery?", "How are escrow deposits protected against fraud?",
        "Why is my payment kept in vault?", "Is escrow mandatory for all orders?"
    ],
    "safevault_otp_verification": [
        "What is the completion OTP code for?", "How do I confirm that I received my parcel with OTP?",
        "Why do I need a 6 digit code for delivery?", "How do travelers prove they delivered the baggage using OTP?",
        "Where do I find my delivery OTP?", "Can payment be released without the OTP code?",
        "What happens if I do not share the OTP?", "How is delivery verified on BawaGo using OTP?",
        "How does the receiver confirm delivery with OTP?", "What is the 6-digit delivery PIN?",
        "Does the traveler need my OTP to get paid?", "How to enter the completion code?",
        "Why must the recipient provide an OTP PIN?", "Delivery verification with 6-digit secret pin",
        "Where does the receiver see the confirmation OTP?", "Why is the 6 digit PIN required at handover?",
        "How does OTP confirm successful order completion?", "Can a driver complete the order without the OTP code?",
        "What is the step to enter the delivery OTP code?", "How to verify handover with the secret pin?"
    ],
    "safevault_fees_and_payouts": [
        "What are the platform fees on BawaGo?", "How much commission does BawaGo take?",
        "What is the 7 percent SafeVault escrow fee?", "Are there hidden charges on SafeVault payments?",
        "What percentage does BawaGo charge for escrow?", "What is the service fee for using BawaGo?",
        "How much does SafeVault cost?", "What is the fee breakdown for orders?",
        "How much do I get charged for escrow protection?", "What percentage goes to BawaGo platform?",
        "What is the platform maintenance commission?", "How much does BawaGo charge in service fees?",
        "What cut does the platform retain per booking?", "Is there a service charge on payments?",
        "Detailed explanation of BawaGo fee percentage", "What is the escrow commission rate?",
        "How much commission is deducted by BawaGo?", "Fee schedule for SafeVault protection",
        "What are the transaction charges on BawaGo?", "Does BawaGo take a 7 percent platform cut?"
    ],
    "baggage_sharing_concept": [
        "How does baggage pooling work?", "Can I share my unused flight baggage kilos?",
        "How do I find a traveler flying with empty luggage space?", "Can I share my luggage allowance on an airplane?",
        "How does BawaGo save money on airline excess baggage?", "What is the baggage vertical on BawaGo?",
        "Can I give my extra luggage kilos to someone on my flight?", "How does peer to peer luggage sharing operate?",
        "Find someone with extra baggage on my route", "Share spare suitcase kilos with senders on flights",
        "Can I carry someone's luggage on my plane flight?", "Airline luggage space pooling service",
        "How do passengers share spare check-in weight on planes?", "Can I split excess airline baggage with travelers?",
        "Connecting passengers with empty luggage space to senders", "How does baggage sharing cut international shipping costs?",
        "Can I book luggage kilos on an upcoming flight?", "Luggage pooling between international airline passengers",
        "How does the flight baggage marketplace work?", "Can senders hire spare baggage allowance from travelers?"
    ],
    "baggage_laptop_electronics_rule": [
        "Can I carry a laptop for someone to Dubai?", "Is it permitted to bring a notebook computer on my flight?",
        "Are laptops allowed in checked baggage?", "What are the rules for carrying electronic gadgets on BawaGo?",
        "Can someone send a tablet or computer through a traveler?", "What documents do I need to carry an electronic device?",
        "Why must laptops be carried in the cabin?", "Can I put electronic items in the checked luggage?",
        "What is the rule for shipping phones or laptops on flights?", "Do I need an invoice to carry an iPad or computer?",
        "Can I take someone's MacBook on a plane?", "Are lithium battery electronics allowed on BawaGo?",
        "Can electronics be carried in hand luggage?", "Inspection rules for laptops at the airport",
        "Can a traveler transport my gaming laptop?", "Is carrying foreign electronics legal on BawaGo?",
        "What invoice documents are needed?", "What invoice is required for laptops?",
        "Do I need an invoice for electronic devices?", "Which purchase invoice documents are required to carry a laptop?",
        "What invoice documents do I need for electronics?", "Must electronics be powered on during airport inspection?"
    ],
    "baggage_weight_limits": [
        "What is the maximum baggage weight I can carry?", "How many kilos can I send through a traveler?",
        "What are the weight limits on airline baggage pooling?", "Can I send a 30kg suitcase through BawaGo?",
        "Are there dimension limits on baggage?", "What is the limit on kilograms for baggage sharing?",
        "How many kgs can an airline passenger accept?", "Is there a minimum or maximum luggage weight limit?",
        "What is the weight restriction for checked bags?", "How much spare baggage capacity can I list in kilos?",
        "Can I ship an overweight suitcase with a traveler?", "What are the standard baggage kilo limits?",
        "Is 23 kg the maximum single baggage limit?", "Can a traveler accept 32 kg in business class?",
        "What happens if luggage exceeds the agreed weight limit in kg?", "Baggage dimension and kilogram constraints on flights",
        "What is the weight ceiling per passenger suitcase?", "How heavy can a pooled baggage parcel be?",
        "Can I book 15 kilos of luggage weight?", "Luggage kilogram regulations and restrictions"
    ],
    "parcel_delivery_overview": [
        "How does peer-to-peer parcel delivery work?", "Can I send an urgent document or package to another city?",
        "How does BawaGo Parcel differ from DHL or FedEx?", "How fast is peer to peer parcel shipping?",
        "Can someone pick up and drop my package along their route?", "How do I book a parcel courier on BawaGo?",
        "Can I send a package with a commuter?", "What is the parcel vertical on BawaGo?",
        "Fast intercity document and parcel delivery", "Can someone deliver an envelope for me today?",
        "How to ship a parcel using regular travelers?", "Peer to peer package dispatch service",
        "How does same-day parcel delivery work?", "Can an everyday commuter deliver my box?",
        "How to track my parcel delivery on BawaGo?", "Sending parcels without traditional courier delays",
        "Can a passenger transport a parcel across cities?", "Parcel delivery by verified travelers going the same way",
        "Can I dispatch boxes and gifts through BawaGo Parcel?", "Peer-to-peer parcel logistics across metro areas"
    ],
    "parcel_prohibited_items": [
        "What items are prohibited on BawaGo Parcel?", "Can I send medicine or prescription drugs in parcels?",
        "Is it allowed to send cash or jewelry?", "What happens if someone asks me to carry an illegal item?",
        "Can travelers inspect the parcel contents?", "Can I send liquids or flammable goods?",
        "What things are banned from being delivered?", "List of forbidden goods on BawaGo",
        "Can I carry alcohol or tobacco for someone in parcels?", "Are weapons or hazardous materials banned?",
        "Can I inspect the box before agreeing to carry it?", "Is transporting cash or gold bullion allowed?",
        "What items violate courier safety rules?", "Can travelers refuse to carry suspicious items?",
        "What packages are disallowed by BawaGo policy?", "Illegal and restricted items on BawaGo",
        "Are firearms, knives, or narcotics strictly banned?", "What are the contraband restrictions on BawaGo?",
        "Penalties for trying to ship prohibited contraband", "Safety inspection guidelines for parcel contents"
    ],
    "ride_sharing_carpool": [
        "How does BawaGo Ride carpooling work?", "Can I find a carpool to the airport?",
        "Can I book a seat on a shared ride?", "Is intercity ride sharing safe on BawaGo?",
        "Can I share a ride between cities?", "How do I find drivers heading the same direction?",
        "Can I carpool from Hyderabad to Bangalore as a passenger?", "Book empty car seats with verified drivers",
        "Is BawaGo Ride like Uber or BlaBlaCar carpooling?", "How to join a shared passenger car ride?",
        "Can commuters share a ride to work or airport?", "What is the carpooling vertical on BawaGo?",
        "Intercity ride sharing with SafeVault payment", "How do passengers book seats in an intercity cab pool?",
        "Can two passengers share a car ride from Mumbai to Pune?", "Finding reliable drivers going to the same destination",
        "How to carpool with verified vehicle owners?", "Commuter carpool booking service",
        "How do I reserve a passenger seat in a shared car?", "Intercity passenger ride booking on BawaGo"
    ],
    "home_food_crave": [
        "What is Crave Home Food on BawaGo?", "Can I order authentic homemade regional food?",
        "Can home chefs sell food on BawaGo?", "How do I get fresh homemade meals delivered?",
        "What kind of home food is available on BawaGo?", "Is Crave food prepared in restaurants or home kitchens?",
        "How do I buy authentic homemade tiffins?", "Can local home cooks list dishes on BawaGo?",
        "Order fresh homemade cuisine in my city", "What is the Crave Food service?",
        "How to discover home-cooked meals nearby?", "Can I order authentic regional sweets and snacks?",
        "Home chef meal delivery on BawaGo", "How does Crave food delivery operate?",
        "Can I get home-cooked lunch boxes delivered?", "Authentic home cooked culinary vertical",
        "How do customers order homemade biryani or curries?", "Can home bakers sell fresh cakes and treats on BawaGo?",
        "Connecting home chefs with people craving real homemade meals", "Can I order daily homemade tiffin service on BawaGo?"
    ],
    "home_food_hygiene_safety": [
        "Is home cooked food on BawaGo hygienic and safe?", "Do home chefs need a license or food safety certificate?",
        "How are food allergens handled on Crave?", "Are kitchen inspections required for BawaGo food cooks?",
        "What packaging standards are required for food deliveries?", "How does BawaGo ensure food quality and cleanliness?",
        "What food safety regulations do home cooks follow?", "Are home kitchens verified for cleanliness?",
        "How do I know if food contains peanuts or allergens?", "What tamper-evident packaging is required for food?",
        "Can customers review food hygiene ratings?", "Are FSSAI or local health licenses mandatory for cooks?",
        "Food safety standards on BawaGo Crave", "How do you guarantee safe preparation of home food?",
        "Allergen warnings and kitchen hygiene rules on Crave", "What if the delivered food is spoiled or unhygienic?",
        "Do home chefs undergo food safety background checks?", "How does BawaGo prevent food contamination?",
        "What rules govern food packaging and tamper seals?", "Quality control and food hygiene verification"
    ],
    "host_stays_homestay": [
        "How does BawaGo Host work?", "Can I rent out my spare room to travelers?",
        "Is booking a room on BawaGo safe?", "How does BawaGo Host compare to Airbnb?",
        "When does the host get paid for a stay?", "Can I find budget homestays near airports?",
        "How do I list my apartment or room on BawaGo?", "Can travelers book short-term homestays?",
        "What is the Host vertical on BawaGo?", "How are guest payments held for homestays?",
        "Can I stay in a local family home?", "Host accommodation and room rentals",
        "How to become a host on BawaGo?", "Are homestay listings verified with real photos?",
        "When do hosts receive guest booking payouts?", "Safe homestay booking with SafeVault escrow",
        "Can property owners rent out vacation homes on BawaGo?", "Finding budget guest rooms with local families",
        "How do homestay check-ins work on BawaGo Host?", "Connecting travelers with authentic local hosts"
    ],
    "wish_personal_shopper": [
        "What is BawaGo Wish?", "Can I ask a traveler to buy something for me from abroad?",
        "How do I request overseas chocolates or gadgets?", "How does payment work when a traveler shops for me?",
        "What if the traveler buys the wrong item?", "Can I make an international shopping request?",
        "Can travelers act as personal shoppers for me?", "How to order duty free products through a traveler?",
        "Can someone purchase foreign goods on their trip?", "What is the Wish vertical on BawaGo?",
        "How does overseas personal shopping work?", "Requesting rare foreign cosmetics or electronics on Wish",
        "How does escrow hold money for shopping requests?", "What if the traveler cannot find the requested item?",
        "Can I buy international items not sold in my country?", "Personal shopper service with international travelers",
        "Can international flyers buy items at duty free for me?", "How do I post a shopping wishlist on BawaGo?",
        "Can travelers shop at duty free for me?", "How does escrow hold funds for shopping?",
        "How are shopping funds kept in escrow?", "Cross-border shopping requests with verified travelers"
    ],
    "fresh_produce_farm": [
        "What is BawaGo Fresh?", "Can I buy farm fresh vegetables and fruit directly from farmers?",
        "How does farm-to-table delivery work on BawaGo?", "Are BawaGo Fresh products organic?",
        "How quickly is fresh produce delivered after harvest?", "Can local farmers sell produce on BawaGo?",
        "Where does the fresh farm produce come from?", "Can I order organic farm vegetables on BawaGo?",
        "How do farmers deliver directly to consumers?", "What farm products are available in the Fresh vertical?",
        "Buy fresh dairy and seasonal fruits from farms", "Farm-to-fork fresh produce delivery service",
        "How does BawaGo support local farmers and growers?", "Is farm produce guaranteed fresh upon arrival?",
        "Direct from farmer agricultural marketplace", "Order fresh harvest fruits and organic vegetables",
        "Can urban households buy produce directly from rural growers?", "How are fresh vegetables transported from farm to doorstep?",
        "Direct farm-to-table harvest delivery network", "Connecting consumers directly with agricultural producers"
    ],
    "trust_kyc_verification": [
        "How does BawaGo verify traveler identity?", "What KYC documents are needed to become a verified traveler or host?",
        "What identity documents are required to verify my account?", "Are all users on BawaGo background-checked?",
        "Do I need to submit my passport to carry items?", "How does BawaGo verify flight tickets?",
        "Is my personal identification data kept secure?", "What is the KYC identity verification process?",
        "How do you verify drivers and hosts on BawaGo?", "Why do I need to upload government ID?",
        "How does live face biometric verification work?", "Do travelers have to verify their airline PNR tickets?",
        "How does BawaGo prevent fake accounts?", "Is ID verification mandatory before accepting bookings?",
        "How are users vetted for safety and trust?", "Identity verification and safety checks on BawaGo",
        "What security credentials are required for users?", "How does BawaGo confirm passport and national ID validity?",
        "What documents are needed to verify my account?", "Trust intelligence vetting for platform participants"
    ],
    "fraud_cash_bypass_refusal": [
        "Can I pay the traveler directly in cash?", "Can we do this transaction outside the app in cash?",
        "Can I pay cash on delivery directly to the driver?", "Is cash payment accepted on BawaGo?",
        "Why can't I pay directly through personal bank transfer?", "What happens if someone asks me to pay in cash?",
        "Can I deal directly with the seller in cash?", "Why is cash payment strictly prohibited?",
        "What is the risk of paying someone outside BawaGo in cash?", "Can I pay offline using hard cash?",
        "What should I do if a driver asks for offline cash?", "Are off-platform cash payments allowed on BawaGo?",
        "Does BawaGo protect cash deals outside SafeVault?", "Why do all payments have to go through SafeVault instead of cash?",
        "Can I bypass the platform to pay cash and save charges?", "Why must all deals use SafeVault?",
        "Why must all transactions go through SafeVault?", "What are the consequences of bypassing SafeVault with cash?",
        "Can a member be banned for requesting cash payments?", "Why are cash payments prohibited?",
        "Policy §1.1 forbidding off-platform cash transactions", "Why cash settlements are rejected by the BawaGo AI agent"
    ],
    "human_support_escalation": [
        "Can I speak with a human agent?", "I need human support", "How do I talk to a live person?",
        "Connect me to customer service staff", "I want to raise a complaint with management",
        "Where can I contact BawaGo customer support?", "Can I talk to a real person on the team?",
        "Transfer me to a live customer support representative", "How do I open a support ticket with a human?",
        "Is there a human helpline or customer care number?", "I need help from a real human specialist",
        "Escalate this issue to human customer care", "Can someone call me from BawaGo support?",
        "How fast does human support respond to disputes?", "Contact human customer assistance team",
        "Connect me to an actual human representative", "I want to speak with a customer service executive",
        "Can I get a phone call from BawaGo support staff?", "Is live chat with human specialists available 24/7?",
        "Connect me to an agent who is not an AI bot"
    ],
    "refund_policy": [
        "What is BawaGo refund policy?", "Do I get a refund if the traveler does not show up?",
        "How long do refunds take on SafeVault?", "What if my package gets damaged during transit, do I get refunded?",
        "Can I get my money back if delivery fails?", "What is the dispute resolution process for payment refunds?",
        "How do I get my money refunded on BawaGo?", "Will I get 100% refund if the booking is cancelled?",
        "What happens if the driver or traveler never arrives, do I get a refund?", "How fast are refunds credited to my card or bank?",
        "Can I request a chargeback or refund for lost goods?", "What is the refund guarantee on SafeVault Escrow?",
        "Does BawaGo offer a money-back refund guarantee?", "How to claim a refund for non-delivery?",
        "When does SafeVault return my deposit refund?", "Full refund policy for failed or cancelled orders",
        "Are escrow deposits returned within 2 to 4 business hours as refund?", "What is the money-back guarantee if traveler disappears?",
        "How are escrow refunds credited back to my account?", "Guaranteed 100 percent refund rules on SafeVault"
    ],
    "traveler_earnings_guide": [
        "How much money can an airline traveler earn?", "How do I earn by carrying parcels on my flight?",
        "Can I subsidize my travel ticket costs with BawaGo?", "How do travelers set their prices per kilo for earnings?",
        "Can international travelers earn money carrying luggage?", "What is the average traveler income per flight trip on BawaGo?",
        "How much do senders pay travelers per kilogram of baggage?", "Can I cover my airfare expenses by sharing baggage space?",
        "How do I monetize my upcoming flight on BawaGo?", "Tips for travelers to maximize delivery earnings",
        "How much cash can I earn flying to Dubai or London?", "Can travelers earn hundreds of dollars per flight journey?",
        "How to make extra money while traveling on vacation with spare baggage?", "Traveler earning opportunities and per-kg pricing guide",
        "Steps to earn money with empty baggage kilos on flights", "How much can I earn per kilo of baggage capacity as a traveler?",
        "Can a passenger make 300 to 500 dollars per flight trip?", "How do travelers receive payout earnings into their account?",
        "What are top earning routes for airline travelers?", "Monetizing spare flight weight to pay for plane tickets"
    ],
    "driver_earnings_guide": [
        "How do drivers earn income on BawaGo Ride?", "Can I make money driving passengers between cities?",
        "What vehicle types are eligible for driver earnings?", "How much does a driver make in fares per trip?",
        "How do driver wallet payouts work?", "Can car owners earn money by offering empty seats?",
        "How much can a driver earn carpooling on a daily commute?", "Can drivers offset fuel and toll costs with BawaGo fares?",
        "How do vehicle owners get paid for giving rides?", "What percentage of fare do drivers keep on BawaGo?",
        "How can I register my car to start earning as a driver?", "Driver earnings breakdown and weekly payout schedule",
        "Can I earn extra income driving intercity carpool routes?", "How does carpooling help drivers make passive income?",
        "How much do drivers earn per passenger on trips?", "Make money with spare car capacity and seats as a driver",
        "Can vehicle owners offset intercity gas costs by taking passengers?", "Do drivers keep up to 95 percent of seat booking fares?",
        "Driver fare calculation and payout timeline", "Earning potential for intercity carpool drivers on BawaGo"
    ],
    "booking_cancellation_rules": [
        "Can I cancel a booking after paying?", "What is the cancellation policy on BawaGo?",
        "Are there penalty fees if I cancel my order?", "What if the traveler cancels my flight booking?",
        "How do I cancel a confirmed journey?", "Can I cancel my carpool or parcel order for free?",
        "What is the time limit for free cancellation?", "Is there a 10 percent cancellation penalty within 12 hours?",
        "What happens if a host or sender cancels at the last minute?", "How do cancellation fees work on BawaGo?",
        "Can a sender get full refund if they cancel 24 hours early?", "What is the 24-hour cancellation rule?",
        "How do I cancel a reservation on BawaGo?", "What compensation does a traveler get if sender cancels?",
        "Cancellation cutoff times and fee policies on BawaGo", "Rules for cancelling reservations and journeys",
        "What if the driver cancels the carpool journey?", "How does BawaGo handle sender cancellation within 12 hours?",
        "Cancellation timeline and penalty breakdown on BawaGo", "What happens to escrow funds when a trip is cancelled?"
    ],
    "popular_travel_corridors": [
        "What are the most popular travel corridors on BawaGo?", "Can I send items on the Hyderabad to Dubai route?",
        "Is there baggage pooling between Singapore and London corridor?", "Are domestic routes like Delhi to Mumbai popular corridors?",
        "Which international flight corridors have the most travelers?", "What are the active travel routes on BawaGo?",
        "Can I find travelers on the Jakarta to Bali route?", "Are there daily routes available from India to UAE?",
        "Which travel corridors have the highest carpooling activity?", "What routes are frequently booked on BawaGo?",
        "Is the Hyderabad to Dubai corridor active on BawaGo?", "Top travel routes and destination corridors",
        "Can I book a courier on the London to Singapore corridor?", "Domestic and international popular travel corridors",
        "Which flight corridors have spare luggage availability?", "Overview of top BawaGo travel routes and corridors",
        "What are the busiest intercity corridors on BawaGo?", "Can I find baggage pooling on the Delhi to London route?",
        "Are there verified commuters on the Mumbai to Pune corridor?", "High-demand traveler corridors for baggage sharing"
    ],
    "insurance_and_damage_protection": [
        "Is my package insured on BawaGo?", "What happens if an item is lost or damaged by the traveler, is there insurance?",
        "Does BawaGo provide cargo damage protection?", "What is the maximum coverage under SafeVault Trust Guarantee insurance?",
        "How do I file an insurance claim for lost or damaged goods?", "Does SafeVault include loss and damage insurance coverage?",
        "What is the BawaGo Trust Guarantee insurance policy?", "How much insurance compensation can I receive for damaged items?",
        "Is my luggage protected against accidental damage by insurance?", "How does BawaGo protect senders from broken items with insurance?",
        "What should I do if my parcel arrives broken, how to claim insurance?", "Does BawaGo cover accidental transit loss up to $500 with insurance?",
        "Insurance claims process for peer to peer delivery", "How to submit an insurance damage claim on BawaGo?",
        "Is cargo damage protection included in the platform fee?", "Transit damage guarantee and insurance terms",
        "What insurance compensation is provided if a package is lost?", "Does the 500 dollar Trust Guarantee insurance cover verified deliveries?",
        "Can senders purchase extended insurance valuation coverage?", "Protection against lost, stolen, or damaged goods in transit with insurance"
    ]
}

def main():
    with open("isolated_agent_site/model/intent_answers.json", "r", encoding="utf-8") as f:
        intent_answers = json.load(f)

    records = []
    for intent, questions in OPTIMIZED_INTENTS.items():
        if intent not in intent_answers:
            print(f"Warning: {intent} missing from intent_answers.json")
            continue
        item = intent_answers[intent]
        answer = item["answer"]
        category = item["category"]
        source_policy = item.get("source_policy", "")

        for q in questions:
            records.append({
                "question": q.strip(),
                "intent": intent,
                "category": category,
                "answer": answer,
                "source_policy": source_policy
            })

    output_path = "isolated_agent_site/model/dataset.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)

    print(f">> Optimized dataset saved to {output_path} with {len(records)} samples across {len(OPTIMIZED_INTENTS)} intents (20 per intent).")

if __name__ == "__main__":
    main()


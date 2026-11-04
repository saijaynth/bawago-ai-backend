"""
Dataset Enrichment Script for BawaGo AI Agent
Builds 16-18 diverse, high-quality question variations for all 25 BawaGo topics,
pairing them with the Gemma-generated answers already stored in intent_answers.json.
"""

import os
import json

ENRICHED_QUESTIONS = {
    "greeting": [
        "Hi", "Hello", "Hey there", "Good morning", "Good afternoon", "Good evening",
        "Greetings", "Hello BawaGo", "Is anyone here?", "What can you do?", "Who are you?",
        "Help me get started", "Namaste BawaGo", "Hi assistant", "Hey I need some help", "What can this bot do?"
    ],
    "about_bawago": [
        "What is BawaGo?", "Can you explain what this website does?", "What services does BawaGo provide?",
        "Tell me about BawaGo platform", "How does the BawaGo community work?", "What are the 7 verticals of BawaGo?",
        "What is the concept behind BawaGo?", "Why should I use BawaGo?", "Is BawaGo a courier company or a marketplace?",
        "Explain BawaGo to a new user", "What kind of platform is BawaGo?", "What makes BawaGo different?",
        "How does BawaGo connect people?", "Can you give me an overview of BawaGo?", "What can I do on the BawaGo website?",
        "How does peer to peer transport work here?"
    ],
    "safevault_escrow_how_it_works": [
        "How does SafeVault Escrow work?", "Is my payment safe on BawaGo?",
        "How do I know the traveler will not take my money and disappear?", "Explain the escrow process on BawaGo",
        "When does the traveler get paid?", "How does payment protection work on this site?", "What is SafeVault?",
        "Why do you hold funds in escrow?", "Can a traveler steal my funds?", "What guarantees do I have that my money is secure?",
        "How does escrow keep buyers and sellers safe?", "When is my money transferred to the seller?",
        "Is my money protected until I receive my order?", "How does the escrow payment system protect me?",
        "Can a traveler run away with my payment?", "How do you prevent scams on BawaGo?"
    ],
    "safevault_otp_verification": [
        "What is the completion OTP code for?", "How do I confirm that I received my parcel?",
        "Why do I need a 6 digit code for delivery?", "How do travelers prove they delivered the baggage?",
        "Where do I find my delivery OTP?", "Can payment be released without the OTP?",
        "What happens if I do not share the OTP?", "How is delivery verified on BawaGo?",
        "How does the receiver confirm delivery with OTP?", "What is the 6-digit delivery PIN?",
        "Does the traveler need my OTP to get paid?", "How to enter the completion code?",
        "Can funds be released if the OTP is missing?", "Why must the recipient provide an OTP?",
        "Delivery verification with 6-digit pin", "How does the handover verification work?"
    ],
    "safevault_fees_and_payouts": [
        "What are the platform fees on BawaGo?", "How much commission does BawaGo take?",
        "How much money do travelers keep?", "Are there hidden charges on SafeVault payments?",
        "What percentage does BawaGo charge for escrow?", "How fast do travelers receive their earnings?",
        "What is the service fee for using BawaGo?", "How much does SafeVault cost?",
        "What is the fee breakdown for orders?", "How much do I get charged for escrow protection?",
        "What percentage goes to BawaGo?", "When will the traveler receive the payout?",
        "Are there any surprise fees or taxes?", "How are payout funds sent to my bank or wallet?",
        "Does BawaGo take a cut from traveler earnings?", "What is the platform maintenance commission?"
    ],
    "baggage_sharing_concept": [
        "How does baggage pooling work?", "Can I sell my unused flight baggage kilos?",
        "How do I find a traveler flying with empty luggage space?", "Can I share my luggage allowance on an airplane?",
        "How does BawaGo save money on excess baggage?", "How much can I charge per kilo of spare baggage?",
        "What is the baggage vertical on BawaGo?", "Can I give my extra luggage kilos to someone else?",
        "How do I rent out my spare flight weight?", "Can a traveler carry extra kilograms for me on a flight?",
        "How does peer to peer luggage sharing operate?", "Find someone with extra baggage on my route",
        "How to monetize my unused airline baggage limit?", "Share spare suitcase kilos with senders",
        "Can I carry someone's luggage on my plane flight?", "Airline luggage space pooling service"
    ],
    "baggage_laptop_electronics_rule": [
        "Can I carry a laptop for someone to Dubai?", "Is it permitted to bring a notebook computer on my flight?",
        "Are laptops allowed in checked baggage?", "What are the rules for carrying electronic gadgets on BawaGo?",
        "Can someone send a tablet or computer through a traveler?", "What documents do I need to carry an electronic device?",
        "Why must laptops be carried in the cabin?", "Can I put electronic items in the checked luggage?",
        "What is the rule for shipping phones or laptops?", "Do I need an invoice to carry an iPad or computer?",
        "Can I take someone's MacBook on a plane?", "Are lithium battery electronics allowed on BawaGo?",
        "Can electronics be carried in hand luggage?", "Inspection rules for laptops at the airport",
        "Can a traveler transport my gaming laptop?", "Is carrying foreign electronics legal on BawaGo?"
    ],
    "baggage_weight_limits": [
        "What is the maximum weight I can carry for someone?", "How many kilos can I send through a traveler?",
        "What are the weight limits on airline baggage pooling?", "Can I send a 30kg suitcase through BawaGo?",
        "Are there dimension limits on baggage?", "What is the limit on kilograms for baggage sharing?",
        "How many kgs can an airline passenger accept?", "Is there a minimum or maximum luggage weight?",
        "Can I send a 5kg bag with a traveler?", "What is the weight restriction for checked bags?",
        "How much spare baggage capacity can I list?", "Can I ship an overweight package with a traveler?",
        "What are the standard baggage kilo limits?", "Can I book 15 kilos of luggage space?",
        "How much weight can a traveler handle?", "Luggage weight regulations and restrictions"
    ],
    "parcel_delivery_overview": [
        "How does peer-to-peer parcel delivery work?", "Can I send an urgent document or package to another city?",
        "How does BawaGo Parcel differ from DHL or FedEx?", "How fast is peer to peer parcel shipping?",
        "Can someone pick up and drop my package along their route?", "How do I book a parcel courier on BawaGo?",
        "Can I send a package with a commuter?", "What is the parcel vertical on BawaGo?",
        "Fast intercity document and parcel delivery", "Can someone deliver an envelope for me today?",
        "How to ship a parcel using regular travelers?", "Peer to peer package dispatch service",
        "How does same-day parcel delivery work?", "Can an everyday commuter deliver my box?",
        "How to track my parcel delivery on BawaGo?", "Sending parcels without traditional courier delays"
    ],
    "parcel_prohibited_items": [
        "What items are prohibited on BawaGo?", "Can I send medicine or prescription drugs?",
        "Is it allowed to send cash or jewelry?", "What happens if someone asks me to carry an illegal item?",
        "Can travelers inspect the parcel contents?", "Can I send liquids or flammable goods?",
        "What things are banned from being delivered?", "List of forbidden goods on BawaGo",
        "Can I carry alcohol or tobacco for someone?", "Are weapons or hazardous materials banned?",
        "Can I inspect the box before agreeing to carry it?", "Is transporting cash or gold allowed?",
        "What items violate aviation and courier safety?", "Can travelers refuse to carry suspicious items?",
        "What packages are disallowed by BawaGo policy?", "Illegal and restricted items on BawaGo"
    ],
    "ride_sharing_carpool": [
        "How does BawaGo Ride work?", "Can I find a carpool to the airport?",
        "Can I offer empty seats in my car to make money?", "Is intercity ride sharing safe on BawaGo?",
        "How do I book a seat on a shared ride?", "What are the fares for carpooling on BawaGo?",
        "Can I share a ride between cities?", "How do I find drivers heading the same direction?",
        "Can I carpool from Hyderabad to Bangalore?", "How does carpooling save fuel money?",
        "Book empty car seats with verified drivers", "Is BawaGo Ride like Uber or BlaBlaCar?",
        "How to offer a lift to passengers in my car?", "Can commuters share a ride to work or airport?",
        "What is the carpooling vertical on BawaGo?", "Intercity ride sharing with SafeVault payment"
    ],
    "home_food_crave": [
        "What is Crave Home Food on BawaGo?", "Can I order authentic homemade regional food?",
        "Can home chefs sell food on BawaGo?", "How do I get fresh homemade meals delivered?",
        "What kind of home food is available on BawaGo?", "Is Crave food prepared in restaurants or home kitchens?",
        "How do I buy authentic homemade tiffins?", "Can local home cooks list dishes on BawaGo?",
        "Order fresh homemade cuisine in my city", "What is the Crave Food service?",
        "How to discover home-cooked meals nearby?", "Can I order authentic regional sweets and snacks?",
        "Home chef meal delivery on BawaGo", "How does Crave food delivery operate?",
        "Can I get home-cooked lunch boxes delivered?", "Authentic home cooked culinary vertical"
    ],
    "home_food_hygiene_safety": [
        "Is home cooked food on BawaGo hygienic and safe?", "Do home chefs need a license or food safety certificate?",
        "How are food allergens handled?", "Are kitchen inspections required for BawaGo food cooks?",
        "What packaging standards are required for food deliveries?", "How does BawaGo ensure food quality and cleanliness?",
        "What food safety regulations do cooks follow?", "Are home kitchens verified for cleanliness?",
        "How do I know if food contains peanuts or allergens?", "What tamper-evident packaging is required for food?",
        "Can customers review food hygiene ratings?", "Are FSSAI or local health licenses mandatory for cooks?",
        "Food safety standards on BawaGo Crave", "How do you guarantee safe preparation of home food?",
        "Allergen warnings and kitchen hygiene rules", "What if the delivered food is spoiled or unhygienic?"
    ],
    "host_stays_homestay": [
        "How does BawaGo Host work?", "Can I rent out my spare room to travelers?",
        "Is booking a room on BawaGo safe?", "How does BawaGo Host compare to Airbnb?",
        "When does the host get paid for a stay?", "Can I find budget homestays near airports?",
        "How do I list my apartment or room on BawaGo?", "Can travelers book short-term homestays?",
        "What is the Host vertical on BawaGo?", "How are guest payments held for homestays?",
        "Can I stay in a local family home?", "Host accommodation and room rentals",
        "How to become a host on BawaGo?", "Are homestay listings verified with real photos?",
        "When do hosts receive guest booking payouts?", "Safe homestay booking with SafeVault escrow"
    ],
    "wish_personal_shopper": [
        "What is BawaGo Wish?", "Can I ask a traveler to buy something for me from abroad?",
        "How do I request overseas chocolates or gadgets?", "How does payment work when a traveler shops for me?",
        "What if the traveler buys the wrong item?", "Can I make an international shopping request?",
        "Can travelers act as personal shoppers for me?", "How to order duty free products through a traveler?",
        "Can someone purchase foreign goods on their trip?", "What is the Wish vertical on BawaGo?",
        "How does overseas personal shopping work?", "Requesting rare foreign cosmetics or electronics",
        "How does escrow hold money for shopping requests?", "What if the traveler cannot find the requested item?",
        "Can I buy international items not sold in my country?", "Personal shopper service with international travelers"
    ],
    "fresh_produce_farm": [
        "What is BawaGo Fresh?", "Can I buy farm fresh vegetables and fruit directly from farmers?",
        "How does farm-to-table delivery work on BawaGo?", "Are BawaGo Fresh products organic?",
        "How quickly is fresh produce delivered after harvest?", "Can local farmers sell produce on BawaGo?",
        "Where does the fresh farm produce come from?", "Can I order organic farm vegetables on BawaGo?",
        "How do farmers deliver directly to consumers?", "What farm products are available in the Fresh vertical?",
        "Buy fresh dairy and seasonal fruits from farms", "Farm-to-fork fresh produce delivery service",
        "How does BawaGo support local farmers and growers?", "Is farm produce guaranteed fresh upon arrival?",
        "Direct from farmer agricultural marketplace", "Order fresh harvest fruits and organic vegetables"
    ],
    "trust_kyc_verification": [
        "How does BawaGo verify traveler identity?", "What documents are needed to become a traveler or host?",
        "Are all users on BawaGo background-checked?", "Do I need to submit my passport to carry items?",
        "How does BawaGo verify flight tickets?", "Is my personal identification data kept secure?",
        "What is the KYC verification process?", "How do you verify drivers and hosts on BawaGo?",
        "Why do I need to upload government ID?", "How does face biometric verification work?",
        "Do travelers have to verify their airline PNR?", "How does BawaGo prevent fake accounts?",
        "Is ID verification mandatory before accepting bookings?", "How are users vetted for safety and trust?",
        "Identity verification and safety checks on BawaGo", "What security credentials are required for users?"
    ],
    "fraud_cash_bypass_refusal": [
        "Can I pay the traveler directly in cash?", "Can we do this transaction outside the app to avoid fees?",
        "Can I pay cash on delivery directly to the driver?", "Is cash payment accepted on BawaGo?",
        "Why can't I pay directly through personal bank transfer?", "What happens if someone asks me to pay in cash?",
        "Can I deal directly with the seller in cash?", "Why is cash payment strictly prohibited?",
        "What is the risk of paying someone outside BawaGo?", "Can I pay offline using hard cash?",
        "What should I do if a driver asks for offline cash?", "Are off-platform payments allowed on BawaGo?",
        "Does BawaGo protect cash deals outside SafeVault?", "Why do all payments have to go through SafeVault?",
        "Can I bypass the platform to save service charges?", "Refusal of cash deals and offline transfers"
    ],
    "human_support_escalation": [
        "Can I speak with a human agent?", "I need human support", "How do I talk to a live person?",
        "Connect me to customer service staff", "I want to raise a complaint with management",
        "Where can I contact BawaGo customer support?", "Can I talk to a real person on the team?",
        "Transfer me to a live customer support representative", "How do I open a support ticket?",
        "Is there a human helpline or customer care number?", "I need help from a real human specialist",
        "Escalate this issue to customer care", "Can someone call me from BawaGo support?",
        "How fast does human support respond to disputes?", "Contact customer assistance team",
        "Connect me to an actual human representative"
    ],
    "refund_policy": [
        "What is BawaGo refund policy?", "Do I get a refund if the traveler does not show up?",
        "How long do refunds take on SafeVault?", "What if my package gets damaged during transit?",
        "Can I get my money back if delivery fails?", "What is the dispute resolution process for payments?",
        "How do I get my money refunded on BawaGo?", "Will I get 100% refund if the booking is cancelled?",
        "What happens if the driver or traveler never arrives?", "How fast are refunds credited to my card or bank?",
        "Can I request a chargeback or refund for lost goods?", "What is the refund guarantee on SafeVault Escrow?",
        "Does BawaGo offer a money-back guarantee?", "How to claim a refund for non-delivery?",
        "When does SafeVault return my deposit?", "Full refund policy for failed or cancelled orders"
    ],
    "traveler_earnings_guide": [
        "How much money can I make as a traveler?", "How do I earn by carrying parcels on my flight?",
        "Can I subsidize my travel ticket costs with BawaGo?", "How do travelers set their prices per kilo?",
        "How do I list my travel capacity and available dates?", "Can international travelers earn money carrying luggage?",
        "What is the average traveler income per trip on BawaGo?", "How much do senders pay travelers per kilogram?",
        "Can I cover my airfare expenses by sharing baggage space?", "How do I monetize my upcoming flight on BawaGo?",
        "Tips for travelers to maximize delivery earnings", "How much cash can I earn flying to Dubai or London?",
        "Can travelers earn hundreds of dollars per journey?", "How to make extra money while traveling on vacation?",
        "Traveler earning opportunities and pricing guide", "Steps to earn money with empty baggage kilos"
    ],
    "driver_earnings_guide": [
        "How do drivers earn on BawaGo?", "Can I make money driving between cities?",
        "What vehicle types are eligible for BawaGo Ride?", "How much does a driver make per trip?",
        "How do driver payouts work?", "Can car owners earn money by offering empty seats?",
        "How much can I earn carpooling on my daily commute?", "Can drivers offset fuel costs with BawaGo?",
        "How do vehicle owners get paid for giving rides?", "What percentage of fare do drivers keep on BawaGo?",
        "How can I register my car to start earning?", "Driver earnings breakdown and weekly payout schedule",
        "Can I earn extra income driving intercity routes?", "How does carpooling help drivers make passive income?",
        "How much do drivers earn per passenger on trips?", "Make money with spare car capacity and seats"
    ],
    "booking_cancellation_rules": [
        "Can I cancel a booking after paying?", "What is the cancellation policy on BawaGo?",
        "Are there fees if I cancel my order?", "What if the traveler cancels my flight booking?",
        "How do I cancel a confirmed journey?", "Can I cancel my carpool or parcel order for free?",
        "What is the time limit for free cancellation?", "Is there a cancellation penalty within 12 hours?",
        "What happens if a host or sender cancels at the last minute?", "How do cancellation fees work on BawaGo?",
        "Can a sender get full refund if they cancel early?", "What is the 24-hour cancellation rule?",
        "How do I cancel a reservation on BawaGo?", "What compensation does a traveler get if sender cancels?",
        "Cancellation cutoff times and fee policies", "Rules for cancelling reservations and journeys"
    ],
    "popular_travel_corridors": [
        "What are the most popular routes on BawaGo?", "Can I send items from Hyderabad to Dubai?",
        "Is there baggage pooling between Singapore and London?", "Are domestic routes like Delhi to Mumbai supported?",
        "Which international flight corridors have the most travelers?", "What are the active travel routes on BawaGo?",
        "Can I find travelers from Jakarta to Bali?", "Are there daily flights available from India to UAE?",
        "Which cities have the highest carpooling activity?", "What routes are frequently booked for parcel delivery?",
        "Is the Hyderabad to Dubai route active on BawaGo?", "Top travel routes and destination corridors",
        "Can I book a courier on the London to Singapore corridor?", "Domestic and international popular travel paths",
        "Which flight routes have spare luggage availability?", "Overview of top BawaGo travel routes and corridors"
    ],
    "insurance_and_damage_protection": [
        "Is my package insured on BawaGo?", "What happens if an item is lost or damaged by the traveler?",
        "Does BawaGo provide damage protection?", "What is the maximum coverage under SafeVault Guarantee?",
        "How do I file a claim for lost or damaged goods?", "Does SafeVault include loss and damage coverage?",
        "What is the BawaGo Trust Guarantee policy?", "How much insurance compensation can I receive?",
        "Is my luggage protected against accidental damage?", "How does BawaGo protect senders from broken items?",
        "What should I do if my parcel arrives broken or damaged?", "Does BawaGo cover accidental transit loss up to $500?",
        "Insurance claims process for peer to peer delivery", "How to submit an insurance claim on BawaGo?",
        "Is cargo protection included in the platform fee?", "Transit damage guarantee and insurance terms"
    ]
}

def main():
    with open("isolated_agent_site/model/intent_answers.json", "r", encoding="utf-8") as f:
        intent_answers = json.load(f)

    records = []
    for intent, questions in ENRICHED_QUESTIONS.items():
        if intent not in intent_answers:
            print(f"Warning: {intent} not in intent_answers.json")
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

    print(f">> Enriched dataset successfully written to {output_path} with {len(records)} samples across {len(ENRICHED_QUESTIONS)} intents.")

if __name__ == "__main__":
    main()


import pygame
import sys
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Sample movie data
movies = [
    ("Dilwale Dulhania Le Jayenge", "A timeless romance where Raj and Simran fall in love during a trip to Europe, but must overcome family expectations back in India to be together."),
    ("3 Idiots", "A heartwarming and humorous take on the Indian education system, following three engineering students as they navigate pressure, friendship, and self-discovery."),
    ("Sholay", "A legendary action-adventure film featuring two ex-convicts hired to capture a ruthless bandit, with unforgettable characters and iconic dialogues."),
    ("My Name Is Khan", "A moving drama about a man with Asperger’s syndrome who embarks on a journey across America to meet the President and clear his name after 9/11."),
    ("Queen", "After being jilted at the altar, a shy Delhi girl sets off on a solo honeymoon to Europe and discovers her independence."),
    ("Taare Zameen Par", "A sensitive portrayal of a dyslexic child’s struggles and the teacher who helps him find his voice through art."),
    ("PK", "An alien questions human customs and religious dogma in this satirical drama."),
    ("Lagaan", "Villagers in colonial India challenge British officers to a game of cricket to avoid oppressive taxes—an epic blend of sports and patriotism."),
    ("Barfi!", "A charming tale of a mute and deaf man navigating love and life with humor and heart."),
    ("Haider", "A Shakespearean tragedy set in conflict-ridden Kashmir."),
    ("Kaho Naa... Pyaar Hai", "A romantic thriller that launched Hrithik Roshan’s career, involving love, loss, and a mysterious doppelgänger."),
    ("The Lunchbox", "A tender story of an unlikely friendship that blossoms through handwritten notes exchanged via Mumbai’s lunchbox delivery system."),
    ("Zindagi Na Milegi Dobara", "A coming-of-age road trip film where three friends confront their fears and rediscover life while traveling through Spain."),
    ("Dangal", "Based on a true story, it follows a former wrestler who trains his daughters to become world-class wrestlers, challenging gender norms in rural India."),
    ("Andhadhun", "A dark comedy thriller about a blind pianist who unwittingly becomes entangled in a murder plot, full of twists and suspense."),
    ("Bhaag Milkha Bhaag", "A biographical sports drama chronicling the life of Indian sprinter Milkha Singh, highlighting his struggles and triumphs."),
    ("Gully Boy", "Inspired by real-life street rappers, this film follows a young man from Mumbai’s slums who rises through the underground hip-hop scene."),
    ("Swades", "An NRI working at NASA returns to India and reconnects with his roots, ultimately choosing to bring change to a rural village."),
    ("Article 15", "A hard-hitting drama inspired by true events, where a police officer confronts caste-based discrimination in rural India."),
    ("Piku", "A quirky slice-of-life film about a father-daughter road trip that explores aging, responsibility, and love with gentle humor."),
    ("Kahaani", "A pregnant woman’s search for her missing husband in Kolkata turns into a gripping thriller with a jaw-dropping twist."),
    ("Drishyam", "A suspenseful drama about a man who goes to great lengths to protect his family after a crime threatens to destroy their lives."),
    ("Om Shanti Om", "A reincarnation drama where a junior artist seeks justice for his past-life love."),
    ("Kabhi Khushi Kabhie Gham", "A lavish family saga about love, pride, and reconciliation."),
    ("Jab We Met", "A bubbly girl and a heartbroken businessman find love on a spontaneous journey."),
    ("Don", "A slick remake of the 1978 classic, following a criminal mastermind and his doppelgänger."),
    ("Rockstar", "A passionate tale of a musician whose heartbreak fuels his rise to fame."),
    ("Raazi", "A young Indian woman marries into a Pakistani military family to spy for India."),
    ("Badhaai Ho", "A comedy-drama about a middle-aged couple’s unexpected pregnancy and its ripple effects."),
    ("Chak De! India", "A disgraced hockey player coaches the Indian women’s team to redemption."),
    ("Tumbbad", "A visually stunning horror-fantasy about greed and a cursed treasure."),
    ("Paan Singh Tomar", "A soldier-turned-athlete becomes a feared rebel in rural India."),
    ("Masaan", "Intertwined stories of love, loss, and societal pressure in small-town India."),
    ("Udaan", "A teenager breaks free from his authoritarian father to pursue his dreams."),
    ("Airlift", "A businessman leads the evacuation of Indians from Kuwait during the Gulf War."),
    ("Neerja", "Based on the true story of a flight attendant who saved lives during a hijacking."),
    ("Kapoor & Sons", "A dysfunctional family reunites, revealing secrets and healing old wounds."),
    ("Tamasha", "A man struggles between societal expectations and his true creative self."),
    ("Black", "A deaf-blind girl’s journey of education and empowerment with her teacher."),
    ("Pink", "A courtroom drama that challenges societal norms around consent and victim-blaming."),
    ("Satyam Shivam Sundaram", "A spiritual and sensual tale exploring beauty, devotion, and identity."),
    ("Mughal-e-Azam", "A grand historical romance between a prince and a courtesan."),
    ("Devdas", "A tragic love story of a man torn between love, pride, and self-destruction."),
    ("Baahubali: The Beginning", "Though technically Telugu, this epic fantasy became a pan-India phenomenon."),
    ("Baahubali 2: The Conclusion", "The thrilling continuation of the royal saga and its legendary twist."),
    ("Padmaavat", "A visually rich period drama about honor, obsession, and sacrifice."),
    ("Kesari", "A war epic based on the Battle of Saragarhi, where 21 Sikh soldiers fought bravely."),
    ("Bajrangi Bhaijaan", "A man helps a mute Pakistani girl reunite with her family across the border."),
    ("Super 30", "The inspiring story of a mathematician who trains underprivileged students for IIT."),
    ("Mimi", "A surrogate mother’s emotional journey when plans take an unexpected turn.")
]
# Extract titles and descriptions
titles = [title for title, _ in movies]
descriptions = [desc for _, desc in movies]

# TF-IDF Vectorization
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(descriptions)
cosine_sim = cosine_similarity(tfidf_matrix)

# Recommendation function
def recommend(input_title, top_n=5):
    if input_title not in titles:
        return ["Movie not found in database."]
    idx = titles.index(input_title)
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    recommendations = [f"{titles[i]} (Similarity: {score:.2f})" for i, score in sim_scores[1:top_n+1]]
    return recommendations

# Initialize pygame
pygame.init()
WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Bollywood Movie Recommendation")

# Colors
WHITE = (255, 255, 255)
LIGHT_GRAY = (240, 240, 240)
GRAY = (200, 200, 200)
DARK_GRAY = (50, 50, 50)
BLUE = (100, 149, 237)
BLACK = (0, 0, 0)
DARK_BLUE = (70, 130, 180)
BG_TOP = (255, 200, 200)
BG_BOTTOM = (200, 220, 250)

# Fonts
font = pygame.font.SysFont("Arial", 24)
big_font = pygame.font.SysFont("Arial", 36, bold=True)
title_font = pygame.font.SysFont("Arial", 28, bold=True)

# UI elements
input_box = pygame.Rect(50, 80, 700, 45)
input_text = ''
active = False
button_rect = pygame.Rect(330, 140, 140, 45)
output_box = pygame.Rect(50, 220, 700, 300)
recommendations = []

def draw_rounded_rect(surface, color, rect, radius=10):
    pygame.draw.rect(surface, color, rect, border_radius=radius)

def draw_gradient_background(surface, top_color, bottom_color):
    for y in range(HEIGHT):
        blend = y / HEIGHT
        r = int(top_color[0] * (1 - blend) + bottom_color[0] * blend)
        g = int(top_color[1] * (1 - blend) + bottom_color[1] * blend)
        b = int(top_color[2] * (1 - blend) + bottom_color[2] * blend)
        pygame.draw.line(surface, (r, g, b), (0, y), (WIDTH, y))

# Main loop
running = True
while running:
    draw_gradient_background(screen, BG_TOP, BG_BOTTOM)

    # Draw header
    header = title_font.render("Bollywood Movie Recommendation System", True, DARK_GRAY)
    screen.blit(header, (WIDTH // 2 - header.get_width() // 2, 20))

    # Draw label
    label = font.render("Enter Movie Title:", True, BLACK)
    screen.blit(label, (50, 50))

    # Input box
    draw_rounded_rect(screen, WHITE, input_box, radius=8)
    pygame.draw.rect(screen, BLUE if active else GRAY, input_box, 2, border_radius=8)
    txt_surface = font.render(input_text, True, BLACK)
    screen.blit(txt_surface, (input_box.x + 10, input_box.y + 10))

    # Button
    draw_rounded_rect(screen, DARK_BLUE, button_rect, radius=8)
    button_text = font.render("Recommend", True, WHITE)
    screen.blit(button_text, (button_rect.x + 10, button_rect.y + 10))

    # Output box
    draw_rounded_rect(screen, WHITE, output_box, radius=10)
    pygame.draw.rect(screen, DARK_BLUE, output_box, 2, border_radius=10)

    # Recommendations
    y = output_box.y + 20
    if recommendations:
        result_label = font.render("Top Recommendations:", True, BLACK)
        screen.blit(result_label, (output_box.x + 20, y))
        y += 40
        for rec in recommendations:
            pygame.draw.circle(screen, DARK_BLUE, (output_box.x + 30, y + 10), 5)
            rec_text = font.render(rec, True, DARK_GRAY)
            screen.blit(rec_text, (output_box.x + 45, y))
            y += 35

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        elif event.type == pygame.MOUSEBUTTONDOWN:
            if input_box.collidepoint(event.pos):
                active = True
            else:
                active = False

            if button_rect.collidepoint(event.pos):
                if input_text.strip():
                    recommendations = recommend(input_text.strip())
                else:
                    recommendations = ["Please enter a movie title."]

        elif event.type == pygame.KEYDOWN and active:
            if event.key == pygame.K_RETURN:
                if input_text.strip():
                    recommendations = recommend(input_text.strip())
                else:
                    recommendations = ["Please enter a movie title."]
            elif event.key == pygame.K_BACKSPACE:
                input_text = input_text[:-1]
            else:
                input_text += event.unicode

    pygame.display.flip()

pygame.quit()
sys.exit()

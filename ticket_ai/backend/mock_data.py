# ruff: noqa: E501  (this module is mostly French prose; wrapping it would hurt readability)
"""Synthetic ticket extract with the same columns as the real one.

It exists so the app can be demonstrated, and tested, without any real data. It
is **not** meant to be statistically faithful; it is meant to behave like an
extract, including the things the app is built to find:

* a steady growth and a weekly rhythm (so projections have something to project);
* a sudden surge of one symptom (VPN after a client update) in the last weeks;
* a slow ramp on mail problems (Outlook), the kind of weak signal an alert
  should catch before it becomes an incident;
* tone, follow-ups and SLA breaches that worsen together, as they do in practice.

Every name, entity and identifier is invented.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from backend.schema import SLA_TARGET_HOURS, Col, derive_priority

SITES = ["Paris", "Toulouse", "Bordeaux", "Brest", "Rennes", "Cholet", "Limours"]
ENTITIES = ["Entité Nord", "Entité Sud", "Entité Ouest", "Entité Export"]
COMPANIES = ["Société Alpha", "Société Beta", "Société Gamma"]
GBUS = ["GBU Défense", "GBU Aéronautique", "GBU Cyber", "GBU Support"]
CONTACTS = (["Portail", "Email", "Téléphone", "Chat"], [0.45, 0.20, 0.28, 0.07])
RESOLUTION_CODES = (
    [
        "Cause racine corrigée",
        "Contournement appliqué",
        "Résolu par le support",
        "Résolu par l'utilisateur",
        "Aucune action requise",
        "Doublon",
    ],
    [0.30, 0.18, 0.30, 0.10, 0.07, 0.05],
)
#: Probability of an arrival at each hour of the day: office hours dominate.
HOURLY = np.array(
    [0.2, 0.1, 0.1, 0.1, 0.1, 0.3, 0.8, 2.5, 6, 9, 8.5, 6, 3, 5, 7.5, 7, 5.5, 3.5, 1.8, 1, 0.7, 0.5, 0.4, 0.3]
)
HOURLY = HOURLY / HOURLY.sum()

# One entry per incident type. ``symptoms`` are (system, title, body) and ``causes``
# are (root cause, resolution summary): what the LLM columns of a real extract hold.
CATALOG: dict[str, dict] = {
    "Réseau": dict(
        weight=0.14,
        service="Connectivité",
        group="N2 Réseau & Sécurité",
        hours=6.0,
        symptoms=[
            (
                "VPN",
                "Déconnexions fréquentes du VPN",
                "Depuis ce matin le VPN se coupe toutes les dix minutes en télétravail, je dois me reconnecter à chaque fois et je perds mes sessions.",
            ),
            (
                "Wi-Fi",
                "Pas de Wi-Fi dans le bâtiment de {site}",
                "Le Wi-Fi corporate est inaccessible au {floor}e étage du bâtiment de {site}, plusieurs collègues sont aussi touchés.",
            ),
            (
                "Réseau local",
                "Réseau très lent sur le site",
                "L'accès aux partages et aux applications est extrêmement lent depuis hier après-midi, impossible de travailler normalement.",
            ),
        ],
        causes=[
            (
                "Saturation du concentrateur VPN",
                "Capacité du concentrateur VPN augmentée et sessions inactives purgées.",
            ),
            ("Défaut de borne Wi-Fi", "Borne Wi-Fi redémarrée puis remplacée par l'équipe réseau."),
            (
                "Mise à jour du client VPN",
                "Retour à la version précédente du client VPN, correctif en cours de déploiement.",
            ),
        ],
    ),
    "Accès & Comptes": dict(
        weight=0.17,
        service="Identité & accès",
        group="N1 Service Desk",
        hours=1.5,
        symptoms=[
            (
                "Active Directory",
                "Réinitialisation du mot de passe Windows",
                "J'ai oublié mon mot de passe Windows et mon compte est verrouillé après plusieurs tentatives, merci de le réinitialiser.",
            ),
            (
                "MFA",
                "Double authentification impossible sur nouveau téléphone",
                "Je viens de changer de smartphone et je n'arrive plus à valider la double authentification pour me connecter.",
            ),
            (
                "SAP",
                "Demande d'accès au rôle SAP achats",
                "Je dois pouvoir valider des commandes dans SAP, merci d'ajouter le rôle achats à mon profil après accord de mon manager.",
            ),
        ],
        causes=[
            ("Compte verrouillé", "Compte déverrouillé et mot de passe réinitialisé après vérification d'identité."),
            ("Jeton MFA non transféré", "Nouveau facteur MFA enrôlé après contrôle d'identité."),
            ("Droits manquants", "Rôle ajouté au profil après validation du manager."),
        ],
    ),
    "Poste de travail": dict(
        weight=0.12,
        service="Poste de travail",
        group="N2 Support Poste",
        hours=9.0,
        symptoms=[
            (
                "Ordinateur portable",
                "Le portable ne démarre plus",
                "L'ordinateur reste bloqué sur l'écran de démarrage depuis la dernière mise à jour Windows, je ne peux plus accéder à mes fichiers.",
            ),
            (
                "Station d'accueil",
                "Écran externe non détecté sur la station d'accueil",
                "La station d'accueil ne reconnaît plus le deuxième écran, l'affichage clignote puis s'éteint.",
            ),
            (
                "Batterie",
                "Batterie qui se décharge très vite",
                "La batterie de mon portable tombe de 100 à 20 pour cent en moins d'une heure sans usage particulier.",
            ),
        ],
        causes=[
            (
                "Mise à jour Windows défaillante",
                "Mise à jour désinstallée en mode récupération puis reportée par politique de groupe.",
            ),
            ("Pilote graphique obsolète", "Pilote graphique et firmware de la station mis à jour."),
            ("Batterie en fin de vie", "Batterie remplacée sous garantie."),
        ],
    ),
    "Messagerie": dict(
        weight=0.13,
        service="Messagerie & collaboration",
        group="N2 Messagerie",
        hours=3.5,
        symptoms=[
            (
                "Outlook",
                "Outlook bloqué sur tentative de connexion",
                "Outlook reste sur tentative de connexion depuis ce matin, je ne reçois plus aucun mail et je ne peux plus en envoyer.",
            ),
            (
                "Teams",
                "Pas de son dans les réunions Teams",
                "Pendant les réunions Teams personne ne m'entend et je n'entends personne, le casque est pourtant reconnu.",
            ),
            (
                "Calendrier",
                "Calendrier partagé plus visible pour l'équipe",
                "Le calendrier partagé de l'équipe a disparu de nos agendas, nous ne pouvons plus planifier de réunions communes.",
            ),
        ],
        causes=[
            ("Profil Outlook corrompu", "Profil Outlook recréé et boîte aux lettres resynchronisée."),
            ("Périphérique audio mal configuré", "Périphérique audio par défaut corrigé et client Teams réinstallé."),
            (
                "Permissions de calendrier perdues",
                "Permissions du calendrier partagé restaurées par l'équipe messagerie.",
            ),
        ],
    ),
    "Application métier": dict(
        weight=0.13,
        service="Applications métier",
        group="N3 Applications",
        hours=12.0,
        symptoms=[
            (
                "SAP",
                "Transaction SAP en erreur",
                "La validation de commande dans SAP se termine par un dump à chaque tentative, toute l'équipe achats est bloquée.",
            ),
            (
                "Portail RH",
                "Dépôt de note de frais refusé",
                "Le portail RH refuse le dépôt de ma note de frais avec une erreur technique alors que le justificatif est au bon format.",
            ),
            (
                "PLM",
                "Extraction PLM qui expire",
                "L'extraction de la nomenclature dans le PLM expire systématiquement après quelques minutes.",
            ),
        ],
        causes=[
            ("Régression après livraison applicative", "Correctif applicatif déployé en urgence par l'équipe éditeur."),
            ("Limite de taille de pièce jointe", "Paramètre de taille maximale relevé sur le portail."),
            ("Timeout trop court", "Délai d'expiration porté de 5 à 20 minutes sur le serveur applicatif."),
        ],
    ),
    "Logiciel": dict(
        weight=0.10,
        service="Logiciels & licences",
        group="N2 Support Poste",
        hours=6.0,
        symptoms=[
            (
                "Installation logiciel",
                "Installation d'un logiciel de modélisation",
                "J'ai besoin que le logiciel de modélisation soit installé sur mon poste pour un projet qui démarre la semaine prochaine.",
            ),
            (
                "Licence",
                "Licence expirée pour un logiciel de calcul",
                "La licence du logiciel de calcul a expiré et l'outil refuse de démarrer, mon livrable est attendu demain.",
            ),
            (
                "Mise à jour",
                "Échec de mise à jour d'un logiciel",
                "La mise à jour du logiciel échoue avec un code d'erreur et laisse l'application inutilisable.",
            ),
        ],
        causes=[
            ("Demande standard", "Logiciel packagé et déployé par le centre de services."),
            ("Licence non renouvelée", "Licence renouvelée par le gestionnaire de parc."),
            ("Package incompatible", "Package corrigé et redéployé sur le poste."),
        ],
    ),
    "Impression": dict(
        weight=0.07,
        service="Impression",
        group="N1 Service Desk",
        hours=4.0,
        symptoms=[
            (
                "Imprimante",
                "Imprimante du 3e étage hors ligne",
                "L'imprimante du troisième étage est hors ligne depuis hier, plus personne du plateau ne peut imprimer.",
            ),
            (
                "Badge",
                "Déverrouillage par badge impossible sur le copieur",
                "Le copieur ne reconnaît plus mon badge, mes impressions restent en file d'attente.",
            ),
            (
                "Pilote",
                "Impossible d'installer le pilote d'imprimante",
                "L'installation du pilote de l'imprimante échoue avec un message d'autorisation insuffisante.",
            ),
        ],
        causes=[
            ("Serveur d'impression arrêté", "Service du serveur d'impression redémarré."),
            ("Badge non enrôlé", "Badge enrôlé de nouveau dans la solution d'impression."),
            ("Droits d'installation insuffisants", "Pilote déployé via le catalogue logiciel."),
        ],
    ),
    "Sécurité": dict(
        weight=0.07,
        service="Sécurité",
        group="N2 Réseau & Sécurité",
        hours=7.0,
        symptoms=[
            (
                "Messagerie",
                "Courriel suspect demandant mes identifiants",
                "J'ai reçu un message qui ressemble à du phishing me demandant de confirmer mon mot de passe, je n'ai pas cliqué.",
            ),
            (
                "Antivirus",
                "Fichier bloqué par l'antivirus",
                "L'antivirus a bloqué un fichier téléchargé depuis un site partenaire, merci d'analyser s'il est sain.",
            ),
            (
                "Certificat",
                "Certificat expiré sur le portail fournisseurs",
                "Le navigateur affiche un avertissement de certificat expiré sur le portail fournisseurs.",
            ),
        ],
        causes=[
            ("Campagne de phishing", "Message supprimé, expéditeur bloqué et utilisateurs prévenus."),
            ("Faux positif antivirus", "Fichier analysé en bac à sable puis autorisé."),
            ("Certificat non renouvelé", "Certificat renouvelé et déployé sur le portail."),
        ],
    ),
    "Stockage": dict(
        weight=0.06,
        service="Stockage & sauvegarde",
        group="N2 Infrastructure",
        hours=5.5,
        symptoms=[
            (
                "Partage réseau",
                "Lecteur réseau non monté à l'ouverture de session",
                "Le lecteur partagé du projet n'est plus monté quand j'ouvre ma session, je dois le reconnecter à la main.",
            ),
            (
                "OneDrive",
                "Synchronisation OneDrive bloquée",
                "OneDrive reste bloqué à zéro pour cent de synchronisation depuis deux jours.",
            ),
            (
                "Sauvegarde",
                "Restauration d'un fichier supprimé",
                "J'ai supprimé par erreur un dossier de travail hier, pouvez-vous le restaurer depuis la sauvegarde.",
            ),
        ],
        causes=[
            ("Script de connexion en erreur", "Script de connexion corrigé et lecteur remappé."),
            ("Cache OneDrive corrompu", "Cache OneDrive réinitialisé et synchronisation relancée."),
            ("Suppression accidentelle", "Fichiers restaurés depuis la sauvegarde de la veille."),
        ],
    ),
    "Autre": dict(
        weight=0.03,
        service="Autre",
        group="N1 Service Desk",
        hours=3.0,
        symptoms=[
            (
                "Non identifié",
                "Question sur un outil interne",
                "Je cherche comment accéder à un outil interne dont je n'ai pas le lien.",
            )
        ],
        causes=[("Demande d'information", "Information communiquée à l'utilisateur.")],
    ),
}

TONE_SENTENCES = {
    "Courtois": ["Merci d'avance pour votre aide.", "Cordialement, merci de votre retour rapide."],
    "Frustré": [
        "C'est la troisième fois que je signale ce problème, toujours rien.",
        "Je suis bloqué et personne ne m'a rappelé malgré mes relances.",
    ],
    "Urgent": [
        "C'est urgent, j'ai une livraison client qui dépend de cela.",
        "Merci de traiter en priorité, la production est impactée.",
    ],
    "Agressif": [
        "C'est inadmissible, je perds ma journée à cause de vous.",
        "Un service aussi lamentable est inacceptable.",
    ],
}
WORK_NOTES = [
    "Analyse en cours : {cause}.",
    "Contact pris avec l'utilisateur, cause probable : {cause}.",
    "Escalade vers le groupe de niveau supérieur, piste : {cause}.",
]


def planted_events(end: pd.Timestamp) -> list[dict]:
    """Return the two events planted in the data, anchored on ``end``.

    A *surge* (flat multiplier, one symptom) and a slow *ramp* (growing
    multiplier). They are what the alerts and the projection should surface.
    """
    return [
        dict(
            kind="Réseau",
            system="VPN",
            cause="Mise à jour du client VPN",
            start=end - pd.Timedelta(days=21),
            end=end,
            multiplier=3.5,
            ramp=False,
        ),
        dict(
            kind="Messagerie",
            system="Outlook",
            cause="Profil Outlook corrompu",
            start=end - pd.Timedelta(days=56),
            end=end,
            multiplier=2.2,
            ramp=True,
        ),
    ]


def generate_tickets(n: int = 5000, seed: int = 7, end: str | None = None, days: int = 540) -> pd.DataFrame:
    """Generate a synthetic extract.

    Args:
        n: Approximate number of tickets (arrivals are Poisson, so it varies slightly).
        seed: Random seed; the same seed always produces the same extract.
        end: Last day of the extract. Defaults to today.
        days: Length of the history.

    Returns:
        A frame with every column of :class:`~backend.schema.Col`, sorted by
        opening date.
    """
    rng = np.random.default_rng(seed)
    last_day = pd.Timestamp(end).normalize() if end else pd.Timestamp.today().normalize()
    # The extract is pulled mid-afternoon: tickets opened later that day do not exist yet,
    # and the most recent ones are still open, as in a real backlog.
    end_ts = last_day + pd.Timedelta(hours=15)
    index = pd.date_range(last_day - pd.Timedelta(days=days - 1), last_day, freq="D")
    events = planted_events(end_ts)

    weekday = np.array([1.25, 1.10, 1.0, 1.0, 0.9, 0.12, 0.08])[index.dayofweek]
    trend = 1.0 + 0.18 * np.arange(len(index)) / 365.0
    summer = np.where(index.month.isin([7, 8]), 0.7, 1.0)
    daily = weekday * trend * summer
    daily = daily / daily.sum() * n

    kinds = list(CATALOG)
    shares = np.array([CATALOG[k]["weight"] for k in kinds])
    shares = shares / shares.sum()

    rows: list[dict] = []
    for position, day in enumerate(index):
        for kind, share in zip(kinds, shares, strict=True):
            rate = daily[position] * share
            event = next((e for e in events if e["kind"] == kind and e["start"] <= day <= e["end"]), None)
            extra = 0
            if event:
                progress = (day - event["start"]).days / max((event["end"] - event["start"]).days, 1)
                factor = 1 + (event["multiplier"] - 1) * (progress if event["ramp"] else 1)
                extra = int(rng.poisson(rate * (factor - 1)))
            made = [_ticket(rng, day, kind, None, end_ts) for _ in range(int(rng.poisson(rate)))]
            made += [_ticket(rng, day, kind, event, end_ts) for _ in range(extra)]
            rows += [ticket for ticket in made if ticket is not None]

    frame = pd.DataFrame(rows).sort_values(Col.OPENED).reset_index(drop=True)
    frame.insert(0, Col.ID, [f"INC{i:07d}" for i in range(1, len(frame) + 1)])
    return frame


def _ticket(
    rng: np.random.Generator, day: pd.Timestamp, kind: str, event: dict | None, end: pd.Timestamp
) -> dict | None:
    """Build one ticket row (without its ``Numéro``), or ``None`` if it opens after the extract.

    ``event`` concentrates the ticket on the planted symptom and cause, and
    makes it larger in scope and angrier in tone.
    """
    spec = CATALOG[kind]
    if event:
        pool = [s for s in spec["symptoms"] if s[0] == event["system"]] or spec["symptoms"]
        symptom = pool[0]
        cause = next((c for c in spec["causes"] if c[0] == event["cause"]), spec["causes"][0])
    else:
        symptom = spec["symptoms"][rng.integers(len(spec["symptoms"]))]
        cause = spec["causes"][rng.integers(len(spec["causes"]))]
    system, title, body = symptom
    site = str(rng.choice(SITES))
    fmt = dict(site=site, floor=int(rng.integers(1, 6)))

    # Scope drives impact; urgency is independent; priority follows from both.
    scope_p = [0.40, 0.30, 0.20, 0.10] if event else [0.78, 0.13, 0.07, 0.02]
    scope = str(rng.choice(["Utilisateur unique", "Équipe", "Site", "Multi-sites"], p=scope_p))
    impact_p = {"Utilisateur unique": [0.60, 0.35, 0.05], "Équipe": [0.20, 0.60, 0.20]}.get(scope, [0, 0.40, 0.60])
    impact = str(rng.choice(["Low", "Medium", "High"], p=impact_p))
    urgency = str(rng.choice(["Low", "Medium", "High"], p=[0.2, 0.45, 0.35] if event else [0.35, 0.5, 0.15]))
    priority = derive_priority(impact, urgency)

    opened = day + pd.Timedelta(hours=int(rng.choice(24, p=HOURLY)), seconds=int(rng.integers(0, 3600)))
    if opened > end:
        return None
    hours = float(rng.lognormal(np.log(spec["hours"] * {"1 - Critical": 0.5, "2 - High": 0.8}.get(priority, 1.0)), 0.8))
    hours *= 1.6 if event else 1.0
    if rng.random() < 0.02:  # the long tail: waiting on a vendor, a spare part, a silent user
        hours *= float(rng.uniform(6, 30))
    resolved = opened + pd.Timedelta(hours=hours)
    closed = resolved + pd.Timedelta(hours=float(rng.uniform(24, 96)))
    is_resolved, is_closed = resolved <= end, closed <= end

    elapsed = (min(closed, end) - opened).total_seconds() / 3600
    breached = (hours if is_resolved else elapsed) > SLA_TARGET_HOURS[priority]
    sla = "Dépassé" if breached else ("Respecté" if is_resolved else "En cours")
    reopen = int(rng.poisson(0.05 + 0.08 * (urgency == "High") + 0.08 * breached)) if is_resolved else 0
    callers = int(rng.poisson(0.3 + min(hours, 72) / 40 + 0.5 * breached))
    support = int(rng.poisson(0.15 + min(hours, 72) / 80))

    # Tone worsens with follow-ups, breaches and, in an event, with the context.
    p_negative = min(0.04 + 0.08 * callers + 0.18 * breached + (0.2 if event else 0), 0.85)
    if rng.random() < p_negative:
        tone = str(rng.choice(["Frustré", "Urgent", "Agressif"], p=[0.5, 0.35, 0.15]))
    else:
        tone = str(rng.choice(["Neutre", "Courtois"], p=[0.6, 0.4]))
    if kind == "Autre":
        request = "Demande d'information"
    elif "Installation" in title or rng.random() < 0.05:
        request = "Demande de service"
    else:
        request = "Incident"

    text = body.format(**fmt)
    if rng.random() < 0.5:
        text = "Bonjour, " + text[0].lower() + text[1:]
    if tone in TONE_SENTENCES:
        text += " " + str(rng.choice(TONE_SENTENCES[tone]))

    if is_resolved:
        stage = "Clos" if is_closed else "Résolu"
    else:
        stage = str(rng.choice(["Nouveau", "Affecté", "En cours", "En attente utilisateur"], p=[0.1, 0.25, 0.5, 0.15]))

    return {
        Col.URGENCY: urgency,
        Col.SERVICE: spec["service"],
        Col.IMPACT: impact,
        Col.PRIORITY: priority,
        Col.STAGE: stage,
        Col.OPENED: opened,
        Col.RESOLVED: resolved if is_resolved else pd.NaT,
        Col.CLOSED: closed if is_closed else pd.NaT,
        Col.TITLE: title.format(**fmt),
        Col.DESCRIPTION: text,
        Col.GROUP: spec["group"],
        Col.ENTITY: str(rng.choice(ENTITIES)),
        Col.RESOLUTION_HOURS: round(hours, 2) if is_resolved else np.nan,
        Col.SLA: sla,
        Col.ELAPSED_HOURS: round(elapsed, 2),
        Col.WORK_NOTES: WORK_NOTES[rng.integers(3)].format(cause=cause[0].lower()) if rng.random() < 0.8 else "",
        Col.COMMENT: "L'utilisateur confirme que le problème persiste." if rng.random() < 0.2 else "",
        Col.RESOLUTION_CODE: str(rng.choice(RESOLUTION_CODES[0], p=RESOLUTION_CODES[1])) if is_resolved else "",
        Col.CLOSE_NOTES: cause[1] if is_closed else "",
        Col.REOPEN_COUNT: reopen,
        Col.CALLER_FOLLOWUPS: callers,
        Col.SUPPORT_FOLLOWUPS: support,
        Col.CONTACT_TYPE: str(rng.choice(CONTACTS[0], p=CONTACTS[1])),
        Col.TGI: f"TGI-{int(rng.integers(1, 6)):02d}",
        Col.COMPANY: str(rng.choice(COMPANIES)),
        Col.SITE: site,
        Col.GBU: str(rng.choice(GBUS)),
        Col.INCIDENT_TYPE: kind,
        Col.TONE: tone,
        Col.REQUEST_TYPE: request,
        Col.SCOPE: scope,
        Col.ROOT_CAUSE: cause[0],
        Col.RESOLUTION_SUMMARY: cause[1] if is_resolved else "",
        Col.AFFECTED_SYSTEM: system,
    }

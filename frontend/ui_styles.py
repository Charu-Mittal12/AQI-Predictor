SIDEBAR_NAV_CSS = """
<style>
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0b1220 0%, #09101b 100%);
        border-right: 1px solid rgba(255,255,255,0.06);
    }
    div[data-testid="stSidebarNav"] {
        background: rgba(14,165,233,0.06);
        border: 1px solid rgba(14,165,233,0.12);
        border-radius: 14px;
        padding: 6px 4px;
        margin-bottom: 14px;
    }
    div[data-testid="stSidebarNav"] a {
        display: flex;
        align-items: center;
        padding: 9px 14px;
        border-radius: 10px;
        font-size: 0.88rem;
        font-weight: 600;
        color: #94a3b8 !important;
        text-decoration: none !important;
        transition: all 0.18s ease;
        margin: 2px 0;
    }
    div[data-testid="stSidebarNav"] a:hover {
        background: rgba(56,189,248,0.10);
        color: #7dd3fc !important;
    }
    div[data-testid="stSidebarNav"] a[aria-current="page"] {
        background: rgba(14,165,233,0.15);
        color: #38bdf8 !important;
        border: 1px solid rgba(14,165,233,0.20);
    }
    div[data-testid="stSidebarNav"]::before {
        content: "NAVIGATE";
        display: block;
        color: #475569;
        font-size: 0.68rem;
        font-weight: 700;
        letter-spacing: 0.1em;
        padding: 4px 14px 6px;
    }
</style>
"""
import subprocess
import time

import pytest
import requests
from playwright.sync_api import Page, expect


@pytest.fixture(scope="module", autouse=True)
def streamlit_server():
    # Start streamlit server in background
    process = subprocess.Popen(["streamlit", "run", "dashboard/app.py", "--server.port", "8502", "--server.headless", "true"])
    
    # Wait for it to be ready
    for _ in range(30):
        try:
            resp = requests.get("http://localhost:8502/_stcore/health")
            if resp.status_code == 200:
                break
        except Exception:
            pass
        time.sleep(1)
    
    yield "http://localhost:8502"
    process.terminate()
    process.wait()

@pytest.mark.e2e
def test_dashboard_pages(page: Page, streamlit_server: str):
    # Set window size for screenshot
    page.set_viewport_size({"width": 1280, "height": 900})
    
    pages_to_test = [
        "Command Center",
        "Forecast Map",
        "Point Explorer",
        "Extreme Guidance",
        "Weight & Explainability",
        "Event Replay",
        "Skill & Drift",
        "Pipeline Status",
    ]
    
    import os
    os.makedirs("docs/figures", exist_ok=True)
    
    # Wait for the app to load initially
    page.goto(f"{streamlit_server}/")
    
    # We will navigate using the sidebar links
    for title in pages_to_test:
        print(f"Testing page: {title}")
        page.click(f"text={title}")
        # Wait for Streamlit to finish running
        page.wait_for_function('() => !document.querySelector("[data-testid=\\"stStatusWidget\\"]") || document.querySelector("[data-testid=\\"stStatusWidget\\"]").innerText === ""', timeout=30000)
        time.sleep(3) # Allow plots to render
        
        # Verify NO exception boxes
        exceptions = page.locator(".stException").count()
        assert exceptions == 0, f"Exception found on {title}"
        
        # Check DEMO badge
        expect(page.locator("text=RUNNING IN DEMO MODE").first).to_be_visible()
        
        page.screenshot(path=f"docs/figures/dashboard_{title.replace(' ', '_').replace('&', 'and').lower()}.png", full_page=True)

@pytest.mark.e2e
def test_map_click_and_deep_link(page: Page, streamlit_server: str):
    # Deep link to Point Explorer
    page.goto(f"{streamlit_server}/point_explorer?lat=15.5&lon=73.5")
    page.wait_for_function('() => !document.querySelector("[data-testid=\\"stStatusWidget\\"]") || document.querySelector("[data-testid=\\"stStatusWidget\\"]").innerText === ""', timeout=30000)
    time.sleep(3)
    
    # Verify it loaded the coordinates from query params
    expect(page.locator("text=15.50")).to_be_visible()
    expect(page.locator("text=73.50")).to_be_visible()

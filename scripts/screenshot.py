import asyncio
from playwright.async_api import async_playwright
import os

async def take_screenshots():
    os.makedirs("docs/figures/dashboard", exist_ok=True)
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page()
        
        print("Navigating to Forecast Map...")
        await page.goto("http://localhost:8502", timeout=60000)
        await page.wait_for_timeout(10000)
        await page.screenshot(path="docs/figures/dashboard/01_forecast_map.png")
        print("Saved forecast map.")
        
        print("Navigating to Weight Maps...")
        await page.click("text=Weight Maps")
        await page.wait_for_timeout(10000)
        await page.screenshot(path="docs/figures/dashboard/02_weight_map.png")
        print("Saved weight map.")
        
        await browser.close()

if __name__ == "__main__":
    asyncio.run(take_screenshots())

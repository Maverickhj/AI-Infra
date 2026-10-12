import { test, expect } from "@playwright/test";

for (const viewport of [
  { width: 1440, height: 1000 },
  { width: 390, height: 844 },
]) {
  test.describe(`decoder reading position ${viewport.width}px`, () => {
    test.use({ viewport });

    test("switching operations keeps the controls at the reading position", async ({
      page,
    }) => {
      await page.goto("/#step=decoder&decoderOp=ffn_norm");
      await page.evaluate(() => document.fonts.ready);
      const panel = page.getByLabel("完整微型 decoder", { exact: true });
      const nav = panel.getByLabel("微型模型子步骤");
      await nav.evaluate((element) =>
        window.scrollTo({
          top: scrollY + element.getBoundingClientRect().top - 160,
          behavior: "instant",
        }),
      );
      const top = await nav.evaluate(
        (element) => element.getBoundingClientRect().top,
      );

      for (const [name, step] of [
        ["Final RMSNorm", "final_norm"],
        ["Embedding", "embedding"],
        ["FFN RMSNorm", "decoder"],
        ["CPU backward", "backward"],
        ["Update / resume", "update"],
        ["Gate × Up", "decoder"],
      ]) {
        const button = nav.getByRole("button", { name, exact: true });
        await button.click();
        await expect(button).toHaveAttribute("aria-pressed", "true");
        expect(
          new URLSearchParams(new URL(page.url()).hash.slice(1)).get("step"),
        ).toBe(step);
        await expect
          .poll(async () =>
            Math.abs(
              (await nav.evaluate(
                (element) => element.getBoundingClientRect().top,
              )) - top,
            ),
          )
          .toBeLessThan(2);
      }

      const final = nav.getByRole("button", {
        name: "Final RMSNorm",
        exact: true,
      });
      await final.focus();
      await page.keyboard.press("Enter");
      await expect(panel.getByTestId("decoder-op")).toContainText(
        "Final RMSNorm",
      );
      await expect
        .poll(async () =>
          Math.abs(
            (await nav.evaluate(
              (element) => element.getBoundingClientRect().top,
            )) - top,
          ),
        )
        .toBeLessThan(2);
      await page.reload();
      await expect(panel.getByTestId("decoder-op")).toContainText(
        "Final RMSNorm",
      );
    });

    test("closing source returns focus to the clicked button without leaving the decoder", async ({
      page,
    }) => {
      await page.goto("/#step=decoder&decoderOp=product");
      await page.evaluate(() => document.fonts.ready);
      const panel = page.getByLabel("完整微型 decoder", { exact: true });
      const source = panel.getByRole("button", { name: /查看此子步骤源码/ });
      await source.evaluate((element) =>
        window.scrollTo({
          top: scrollY + element.getBoundingClientRect().top - 300,
          behavior: "instant",
        }),
      );
      const position = await page.evaluate(() => scrollY);
      for (const close of ["escape", "button"]) {
        await source.click();
        await expect(page.getByRole("dialog")).toBeVisible();
        if (close === "escape") await page.keyboard.press("Escape");
        else
          await page
            .getByRole("button", { name: "关闭源码", exact: true })
            .click();
        await expect(page.getByRole("dialog")).not.toBeVisible();
        await expect(source).toBeFocused();
        await expect
          .poll(async () =>
            Math.abs((await page.evaluate(() => scrollY)) - position),
          )
          .toBeLessThan(2);
      }
    });
  });
}

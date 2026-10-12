import { test, expect } from "@playwright/test";

for (const viewport of [
  { width: 1440, height: 1000 },
  { width: 390, height: 844 },
]) {
  test.describe(`shared reading position ${viewport.width}px`, () => {
    test.use({ viewport });

    test("all source entry points restore their own trigger and page position", async ({
      page,
    }) => {
      test.setTimeout(90000);
      for (const [route, region] of [
        ["/#step=decoder", "当前步骤"],
        ["/#step=input&view=sample", "SFT 数据演算"],
        ["/#step=decoder", "GQA 精讲"],
        ["/#step=decoder", "TP DP rank inspector"],
        ["/#step=decoder", "PP SP CP sequence journey"],
        ["/#model=deepseek-v3&step=decoder", "MoE MLA 完整微型模型"],
        ["/#scenario=rl&step=policy_update", "RL 数值参考闭环"],
        ["/#view=runtime", "只读运行对照"],
      ]) {
        await page.goto(route);
        await page.evaluate(() => document.fonts.ready);
        const panel = page.getByLabel(region, { exact: true });
        const trigger = panel
          .getByRole("button", {
            name: /查看.*源码|打开 GQA 子步骤源码|Dataset → collator/,
          })
          .first();
        await trigger.evaluate((element) =>
          window.scrollTo({
            top: scrollY + element.getBoundingClientRect().top - 280,
            behavior: "instant",
          }),
        );
        const top = await trigger.evaluate(
          (element) => element.getBoundingClientRect().top,
        );
        const element = await trigger.elementHandle();
        // Touch/pointer activation may leave activeElement on a different control.
        await trigger.evaluate((element) =>
          element.addEventListener(
            "mousedown",
            (event) => event.preventDefault(),
            { once: true },
          ),
        );
        await trigger.click();
        await expect(page.getByRole("dialog")).toBeVisible();
        const line = page
          .getByRole("dialog")
          .getByRole("button", { name: /^定位 L/ })
          .first();
        if (await line.count()) await line.click();
        expect(
          Math.abs(
            (await element!.evaluate(
              (control) => control.getBoundingClientRect().top,
            )) - top,
          ),
          region,
        ).toBeLessThan(2);
        const bounds = await page.getByRole("dialog").boundingBox();
        if (bounds && (bounds.x > 2 || bounds.y > 2))
          await page.mouse.click(0, 0);
        else await page.keyboard.press("Escape");
        await expect(page.getByRole("dialog")).not.toBeVisible();
        await expect(trigger).toBeFocused();
        expect(
          Math.abs(
            (await trigger.evaluate(
              (control) => control.getBoundingClientRect().top,
            )) - top,
          ),
          region,
        ).toBeLessThan(2);
      }
    });

    test("shared selectors keep focus and their viewport position", async ({
      page,
    }) => {
      test.setTimeout(60000);
      for (const [route, name, value] of [
        ["/#step=decoder", "微型层", "1"],
        ["/#step=decoder", "TP", "1"],
        ["/#step=decoder", "Sequence layout", "thd"],
        [
          "/#model=deepseek-v3&step=decoder&familyOp=mla-cache",
          "Family MLA mode",
          "decode",
        ],
        ["/#scenario=rl&step=policy_update", "RL algorithm", "ppo"],
      ]) {
        await page.goto(route);
        await page.evaluate(() => document.fonts.ready);
        const control = page.getByLabel(name, { exact: name !== "微型层" });
        await control.evaluate((element) =>
          window.scrollTo({
            top: scrollY + element.getBoundingClientRect().top - 180,
            behavior: "instant",
          }),
        );
        await control.focus();
        const top = await control.evaluate(
          (element) => element.getBoundingClientRect().top,
        );
        await control.selectOption(value);
        await expect(control).toHaveValue(value);
        await expect(control).toBeFocused();
        await expect
          .poll(async () =>
            Math.abs(
              (await control.evaluate(
                (element) => element.getBoundingClientRect().top,
              )) - top,
            ),
          )
          .toBeLessThan(2);
      }
    });
  });
}

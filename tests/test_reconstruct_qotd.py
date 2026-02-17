"""
Reconstruct 'Quote of the Day' shortcut from scratch using only the compiler API.
55 actions, 25 unique types, conditionals, menus, repeat-each, variables.
"""
import sys, os
from pathlib import Path
sys.path.insert(0, str(Path(os.path.abspath(__file__)).parent.parent / "src"))
from shortcuts_compiler import (
    Shortcut, actions, ref_extension_input, ref_variable
)

sc = Shortcut("Quote of the Day")

# 0. Comment
sc.add(actions.make("comment"))

# 1-11. First conditional: does shortcut input have any value?
with sc.if_else_block(ref_extension_input(), condition="has_any_value") as otherwise:
    # IF branch: input exists (it's a podcast episode from library)
    pods = sc.add(actions.make("getpodcastsfromlibrary"))
    chosen_show_raw = sc.add(actions.make("choosefromlist",
        WFChooseFromListActionPrompt="Which show?", WFInput=pods))
    sc.add(actions.make("setvariable", WFVariableName="Chosen Show", WFInput=chosen_show_raw))
    episodes = sc.add(actions.make("getepisodesforpodcast", WFInput=chosen_show_raw))
    chosen_ep_raw = sc.add(actions.make("choosefromlist",
        WFChooseFromListActionPrompt="Which episode?", WFInput=episodes))
    sc.add(actions.make("setvariable", WFVariableName="Chosen Episode", WFInput=chosen_ep_raw))
    store_url = sc.add(actions.make("properties.podcast",
        WFContentItemPropertyName="Store URL", WFInput=chosen_ep_raw))
    
    # ELSE branch
    otherwise()
    detected_link = sc.add(actions.make("detect.link", WFInput=ref_extension_input()))

# 12-15. Extract podcast ID from URL
split1 = sc.add(actions.make("text.split", text=store_url,
    WFTextSeparator="Custom", WFTextCustomSeparator="id", **{"Show-text": True}))
last_item = sc.add(actions.make("getitemfromlist", WFInput=split1, WFItemSpecifier="Last Item"))
split2 = sc.add(actions.make("text.split", text=last_item,
    WFTextSeparator="Custom", WFTextCustomSeparator="?", **{"Show-text": True}))
first_item = sc.add(actions.make("getitemfromlist", WFInput=split2))

# 16-19. Search for podcast and get show info
search_result = sc.add(actions.make("searchpodcasts",
    WFAttribute="Product ID", WFSearchTerm=first_item))
show_url_val = sc.add(actions.make("properties.podcastshow",
    WFContentItemPropertyName="Store URL", WFInput=search_result))
sc.add(actions.make("setvariable", WFVariableName="Show URL", WFInput=show_url_val))
show_title = sc.add(actions.make("properties.podcastshow",
    WFContentItemPropertyName="Name", WFInput=search_result, CustomOutputName="Show Title"))

# 20-33. Second conditional
with sc.if_else_block(store_url, condition="has_any_value") as otherwise2:
    sc.add(actions.make("setvariable", WFVariableName="Episode URL", WFInput=store_url))
    article = sc.add(actions.make("getarticle", WFWebPage=store_url))
    article_title = sc.add(actions.make("properties.articles", WFInput=article))
    title_split = sc.add(actions.make("text.split", text=article_title,
        WFTextSeparator="Custom", WFTextCustomSeparator=" on Apple Podcasts", **{"Show-text": True}))
    clean_title = sc.add(actions.make("getitemfromlist", WFInput=title_split))
    sc.add(actions.make("setvariable", WFVariableName="Full Title", WFInput=clean_title))
    
    otherwise2()
    ep_store_url = sc.add(actions.make("properties.podcast",
        WFContentItemPropertyName="Store URL", WFInput=chosen_ep_raw))
    sc.add(actions.make("setvariable", WFVariableName="Episode URL", WFInput=ep_store_url))
    sc.add(actions.make("setvariable", WFVariableName="Title", WFInput=chosen_ep_raw))
    title_text = sc.add(actions.make("gettext", WFTextActionText=chosen_ep_raw))
    sc.add(actions.make("setvariable", WFVariableName="Full Title", WFInput=title_text))

# 34-36. User prompts
timestamp = sc.add(actions.make("ask",
    WFAskActionPrompt="What\u2019s the timestamp?", CustomOutputName="Timestamp"))
quote = sc.add(actions.make("ask",
    WFAskActionPrompt="What\u2019s the quote?", CustomOutputName="Quote"))
speaker = sc.add(actions.make("ask",
    WFAskActionPrompt="Who\u2019s the speaker?", CustomOutputName="Speaker Name"))

# 37-54. Menu: save destination
with sc.menu_block("Where do you want to save it?", ["Day One", "Notes"]) as cases:
    cases["Day One"]()
    split_quote = sc.add(actions.make("text.split", text=quote, **{"Show-text": True}))
    with sc.repeat_each_block(split_quote):
        line_text = sc.add(actions.make("gettext", WFTextActionText=split_quote))
    quoted_text = sc.add(actions.make("text.combine", text=line_text,
        CustomOutputName="Quoted Text", **{"Show-text": True}))
    entry_text = sc.add(actions.make("gettext", WFTextActionText=quoted_text))
    sc.add(actions.make("com.dayonelog.dayoneiphone.post",
        EntryJournal="Personal", EntryTags=speaker, EntryText=entry_text, imageClipboard=quote))
    
    cases["Notes"]()
    note_text = sc.add(actions.make("gettext", WFTextActionText=quote))
    sc.add(actions.make("com.apple.mobilenotes.SharingExtension",
        WFCreateNoteInput=note_text, OpenWhenRun=False, ShowWhenRun=False))
    clipboard_text = sc.add(actions.make("gettext", WFTextActionText=quote))
    rich_text = sc.add(actions.make("getrichtextfrommarkdown", WFInput=clipboard_text))
    sc.add(actions.make("setclipboard", WFInput=rich_text))
    sc.add(actions.make("showresult"))
    sc.add(actions.make("shownote", WFInput=note_text))

import tempfile
filepath = sc.save(os.path.join(tempfile.gettempdir(), "Quote_of_the_Day_reconstructed.shortcut"))
print(f"Saved to: {filepath}")
print(f"Actions: {len(sc.actions)}")
print(f"Unique types: {len(set(a['WFWorkflowActionIdentifier'] for a in sc.actions))}")

from datetime import date
from p_utils import normalize_team
import re
import boto3
import os

current_year = date.today().year

def get_dynamic_min_pa(year):
    # Only apply to current season
    if year == 2020:
        return 186
    if year < current_year:
        return 502

    opening_day = date(current_year, 3, 26) # rough estimate
    today = date.today()

    days_since = (today - opening_day).days
    min_pa = max(10, int(days_since * 2.5))

    return min(min_pa, 502)

def get_dynamic_min_ip(year):
    if year < current_year:
        return 162

    opening_day = date(current_year, 3, 26) # rough estimate
    today = date.today()

    days_since = (today - opening_day).days

    # Approx: 0.8 games per day → 1 IP per game
    min_ip = max(5, days_since * 0.8)

    return min(min_ip, 162)

def get_percentile_min_pa(year):
    if year == 2020:
        return 126
    if year < current_year:
        return 340  

    opening_day = date(current_year, 3, 26) # rough estimate
    today = date.today()
    days_since = (today - opening_day).days

    # 2.1 PA per team game pace → ~340 over full season
    min_pa = max(20, days_since * 2.1)

    return min(min_pa, 340)

def get_percentile_min_ip(year):
    if year == 2020:
        return 15
    if year < current_year:
        return 40  

    opening_day = date(current_year, 3, 26)
    today = date.today()
    days_since = (today - opening_day).days

    # ~0.25 IP per team game → ~40 over season
    min_ip = max(3, days_since * 0.25)

    return min(min_ip, 40)

TEAM_MLB_IDS = {
    "ARI": 109, "ATL": 144, "BAL": 110, "BOS": 111,
    "CHC": 112, "CIN": 113, "CLE": 114, "COL": 115,
    "CHW": 145, "DET": 116, "HOU": 117, "KCR": 118,
    "LAA": 108, "LAD": 119, "MIA": 146, "MIL": 158,
    "MIN": 142, "NYM": 121, "NYY": 147, "OAK": 133,
    "PHI": 143, "PIT": 134, "SDP": 135, "SEA": 136,
    "SFG": 137, "STL": 138, "TBR": 139, "TEX": 140,
    "TOR": 141, "WSN": 120,
}

DIVISION_TEAMS = {
    "AL East":    {"BAL", "BOS", "NYY", "TBR", "TOR"},
    "AL Central": {"CHW", "CLE", "DET", "KCR", "MIN"},
    "AL West":    {"HOU", "LAA", "OAK", "SEA", "TEX"},
    "NL East":    {"ATL", "MIA", "NYM", "PHI", "WSN"},
    "NL Central": {"CHC", "CIN", "MIL", "PIT", "STL"},
    "NL West":    {"ARI", "COL", "LAD", "SDP", "SFG"},
}

ALL_DIVISIONS = list(DIVISION_TEAMS.keys())

def get_team_division(abbrev: str) -> str:
    a = normalize_team(abbrev)
    for div, teams in DIVISION_TEAMS.items():
        if a in teams:
            return div
    return ""

def get_team_logo_url(abbrev: str) -> str:
    mlb_id = TEAM_MLB_IDS.get(normalize_team(abbrev))
    if mlb_id:
        return f"https://www.mlbstatic.com/team-logos/{mlb_id}.svg"
    return ""


BREF_COOKIE = "osano_consentmanager_uuid=6e07dec4-0900-4f61-8bf1-5225101aec40; _ga=GA1.1.1676421375.1773543362; hubspotutk=54ad0caad9ff4eccb1efc2694cf5b0ed; _lc2_fpi=7eda703e725e--01kkqpr8cjpzzeysd2f26sb72a; _lc2_fpi_meta=%7B%22w%22%3A1773543367058%7D; _lr_env_src_ats=false; _pubcid=7ef03e9c-6c47-4b70-92d7-1c01b073b0ca; gamera_user_id=9a1e19b6-0b3d-4c36-9235-11d714b6aeb0; ccuid=e82c26cd-6f68-4e77-8e7c-62cf1e497f01; mm-user-id=HyPVuTOhLfcawcSR; _au_1d=AU1D-0100-001773543367-M6NSQT9L-KPU7; gcid_first=ae8acff5-b6e5-4a9d-bfc6-751e410af850; uuid=83127E57-F135-48EC-9AC4-724C6DA9CB22; _sharedid=86878fa6-4e0b-4ba6-a350-5747fd72660d; hb_insticator_uid=8a8e7091-4c83-46c4-a4a0-c7cec63735d9; _cc_id=d079a59afbd52a3da9161bac7782e186; _sharedid_cst=nCxMLIssQw%3D%3D; pbjs-unifiedid_cst=nCxMLIssQw%3D%3D; id5id_cst=nCxMLIssQw%3D%3D; _ga_FVWZ0RM4DH=GS2.1.s1776438465$o52$g0$t1776438465$j60$l0$h0; pbjs_fabrickId_cst=nCxMLIssQw%3D%3D; __tamLIResolveResult_cst=nCxMLIssQw%3D%3D; meta_more_button=1; pbjs-unifiedid=%7B%22TDID%22%3A%220e2059cd-aef5-4f94-8805-246e889afaac%22%2C%22TDID_LOOKUP%22%3A%22TRUE%22%2C%22TDID_CREATED_AT%22%3A%222026-04-06T16%3A05%3A52%22%7D; fsuid=9d74497d-2cfd-4b4c-a7cd-106bc982fc01; id5id=%7B%22signature%22%3A%22ID5_AnuTs9A_dU22R33WjucWghUR1-yjcXWYNdpHnL5SZgDK5yRh1pgmPXOr94Q1UfyjT2ntZZgosd6tC_Pj9KsBbB6OdyjSQmHPykufaaSFV_i4AYJ3cqboqNedOFfeGUoXhBKsVGkzyiSp_bFEU7QXG-f2V-sFIVnTTwW-v6eq6J1jIv6RbpE%22%2C%22created_at%22%3A%222026-03-15T02%3A52%3A37.9Z%22%2C%22id5_consent%22%3Atrue%2C%22original_uid%22%3A%22ID5*B3Exjsx_D4AibOt7Yzjgtzxp9XcU-2kGlbv3H916uub__2oCLaRaAAEBCmm2HvUAqqM9S8msVfMzWx2owqdsBQ%22%2C%22universal_uid%22%3A%22ID5*T67h_grxyFP6CGk0F6Q_IKdAVMzXOYAobb7Un_1Q5Sj__2oCLaRaAAEBCmm2HvUAqqONfZrYxdnGXEehgBoD1w%22%2C%22link_type%22%3A2%2C%22cascade_needed%22%3Atrue%2C%22privacy%22%3A%7B%22jurisdiction%22%3A%22other%22%2C%22id5_consent%22%3Atrue%7D%2C%22ext%22%3A%7B%22linkType%22%3A2%2C%22pba%22%3A%22RduBatrm5u7EzknLRrZqNkY7AMOtyxvv6hhioJt%2Fh4fagbiFRqtW80bsejLr2GeQTXxuGXN05T3cw8zOGXgxnQ%3D%3D%22%7D%2C%22cache_control%22%3A%7B%22max_age_sec%22%3A7200%7D%2C%22ids%22%3A%7B%22id5id%22%3A%7B%22eid%22%3A%7B%22source%22%3A%22id5-sync.com%22%2C%22uids%22%3A%5B%7B%22id%22%3A%22ID5*T67h_grxyFP6CGk0F6Q_IKdAVMzXOYAobb7Un_1Q5Sj__2oCLaRaAAEBCmm2HvUAqqONfZrYxdnGXEehgBoD1w%22%2C%22atype%22%3A1%2C%22ext%22%3A%7B%22linkType%22%3A2%2C%22pba%22%3A%22RduBatrm5u7EzknLRrZqNkY7AMOtyxvv6hhioJt%2Fh4fagbiFRqtW80bsejLr2GeQTXxuGXN05T3cw8zOGXgxnQ%3D%3D%22%7D%7D%5D%7D%7D%7D%2C%22tags%22%3A%7B%22id%22%3A%22y%22%7D%7D; id5id_last=Mon%2C%2011%20May%202026%2019%3A27%3A32%20GMT; is_live=true; __hssrc=1; _li_dcdm_c=.baseball-reference.com; panoramaId_expiry=1780022588983; panoramaId=d62080c79f29784ab233452539e416d5393844b81fcdaf500ff04877117549fb; panoramaIdType=panoIndiv; sr_n=2%7CSat%2C%2023%20May%202026%2003%3A18%3A10%20GMT; _lr_sampling_rate=100; pbjs_fabrickId=%7B%22fabrickId%22%3A%22E1%3AeKP-kl6Jj_DxUjVZZWhjqxQicmxm6QwkTe2dbsTbUe6L3bVkEF2duYM27fWAjn2ejPCoM6SAXjKAcyZ6ff9hv9Wb9DYxKb5gFP4OXk_D5jtPmKf-2OyDsbKewEQiacWa%22%7D; pbjs_li_nonid=%7B%22nonId%22%3A%2229-QMyfSNI7SQ%2BFNMEbRFli0yyZlWwICCoLVec4aViyOZ5X%2F4VZT9zgylshXLpMmCzCDM%2F3ZVw310%2FykUWIeQJDwgalYQD2GEpR7hD%2F9MfEV2WCCw%3D%3D%22%7D; pbjs_li_nonid_cst=nCxMLIssQw%3D%3D; __hstc=107817757.54ad0caad9ff4eccb1efc2694cf5b0ed.1773543362153.1779819968496.1779853905446.262; __gads=ID=85c3f8237895f3c6:T=1773543368:RT=1779853906:S=ALNI_Mbqn-nqV092Xrd4rRc_JQQ6qVRt_g; __gpi=UID=00001350b0c9be53:T=1773543368:RT=1779853906:S=ALNI_MZkQwoyIKfw_hJWg5btPMiHfpnqlw; __eoi=ID=aead3432f3ffa49b:T=1773543368:RT=1779853906:S=AA-AfjYk5ib9AgBu63VksKcaa_O0; cto_bundle=7HHnTV82Y05tRDY5JTJCMUhRTlVCUE80ZCUyQnhLQ0RxWjBHR2tJWTZtSnR0ZENJTEFPbjNKUk9hQkx5bXkzUWpNcU1Pc2IlMkJXRjB1b2lnUkRNSmQlMkJzMFFYc2xvWXgxZG16bCUyQkRwTDViVjVMaW4xc090WVRnNnEzck5VbGdHMnhLSFpMWHRoTEVManBNZ015eW1nSjIzcmFLU01USFJYTmdJJTJGWjdLSkNyc2JBVUZyMEo5MDJnMGYzOVlJZUVhaUlNRk5mWUV6RWpXVyUyQkp4b1NQTzhtS1ZHa3FHbHpXUFElM0QlM0Q; cto_bidid=s4xLBV9WMGJlZHN5aExoUWdXcjklMkJvT3NOVXk3eDRiVTQxQmpxNWdxWkJ0SEElMkZicUxVTUZuN2VEUm05bnlSWHFCenNpRXZVb0RiJTJCOWljbDRGeE82blRQRUslMkZPWXNJNHdYTGk0bjFxZHpocU0ycGo0RG95bkJjZjhkU2RrSEdDJTJGcjBFOFVraXdBVndFOUNEaDZaZ0xYZVZjUmR6NllUaUhycUp5N2JaWEkwY29ZbVg4JTNE; _ga_9W1J1KYCNH=GS2.1.s1779887754$o281$g0$t1779887754$j60$l0$h0; _ga_80FRT7VJ60=GS2.1.s1779887754$o281$g0$t1779887754$j60$l0$h0; __cf_bm=8o504FR9Lem8E4MWUNeBN3rXg.2H4.KvORqe6cF9ADQ-1779890257.2503767-1.0.1.1-ABgAfMJ6ZRDeoG.VnLEbWw2P6fv9tW5sRRvbHOb32E54ME8ykojvvI9M4SgoWlctysoVEe5CFZjFDj7LoUx7mH5Zqpy93kIYDGi5GxNMNGStnqGqcQopy5VZ3X4X6_ve"
FG_COOKIE = "fg_uuid=2431addc-3cd7-4544-b466-812ec27f44ef; mm-user-id=C7jXGvE0tOoRnNcS; gcid_first=ae8acff5-b6e5-4a9d-bfc6-751e410af850; _au_1d=AU1D-0100-001773754256-730TH4D1-BCK9; _ga=GA1.1.1466717718.1773754396; uuid=49F2CBB5-4945-4F31-9206-77886E17DA0D; cf_clearance=BpYdFP7Z.1NYjllqj5rka4zSYZAHXjRs6f_Wjt8jgrI-1774151784-1.2.1.1-j03U8qlpzMauLZR16Ly_mOkQiRmWLLQ.PBCVpUR6W3EQmQ0WNtwCnG.T7S1PSFP2rZraQVie0yeLFBtrtCnK63Uar6pQKsBrnjyoGU0Z_NF52eA5KyqGA20JT_5G42RNHzoWq.3pqkbt1fA0Mkiyp5YUs5tgT_GMyTVDO54Ae4WzzhS5_TlampcXz1Yl4.KEGRwMrOieqW10nG9ah4PHrrRY19haHtjQkoqyRg29rzs; _pubcid=23e55e4f-3773-4abb-914a-1470dcd28a5c; _cc_id=c42ed0452dd78c58fd557a22eb97c5d8; __gads=ID=fb87d042ac304697:T=1773754256:RT=1774555008:S=ALNI_MbeFf3_w4pFPuBqyUVJnQ3lSYvjSw; __gpi=UID=00001340df0a4670:T=1773754256:RT=1774555008:S=ALNI_MbdoAitB3oFf35DU4YgQbObI0dw-g; __eoi=ID=2951353c22d0bf60:T=1773754256:RT=1774555008:S=AA-Afjave5gW7rGBQnBUy1mx6wgR; wordpress_logged_in_0cae6f5cb929d209043cb97f8c2eee44=rse1%7C1806111939%7CPWE0VFS1wK7FjLbXbQpKLoBC3kuCbDpXVlerOiPzxTj%7Cd68ac69d0a2271f6ad7fa8482722abb9e79cc4c988b48e03520001938738c289; cto_bundle=8CXHYl94VnRtYkR1Zmg5SVJlSFI1THlaMkkwd3lVR2hKYzFJTklsbm02SyUyQk5wNUdBWjVqYnM1VkNlWGZxbnprSkZweE9wNlJlTEg1c3F4R1FGTHM0SUpKWkZPdG00dWtFRzhPclp3M29PY2ZRdVRWQjB5d1V0c3Q1aFV3d0o3UHc4QlduRElhZVFnbHZPbVRnWDBiSE5qQnR4dyUzRCUzRA; cto_bidid=-7o1i18wYUl3d2VJJTJGMVh6JTJCV2h3bUEybVNsZEVKc256T3pCOTE1V0UwQjZuNjBDQkhpMyUyQmhsTkJKeENmWjZvV1FrcXlOWnVhemZYTWY4d2U0WTBXVm1CUjVjWjdzeCUyQmJ4dUgwcG5pQ083OFNCVlVjJTNE; wp_automatewoo_visitor_0cae6f5cb929d209043cb97f8c2eee44=w6gq7ungauuz1hgtzbq4; _ga_FVWZ0RM4DH=GS2.1.s1774555020$o7$g0$t1774555025$j55$l0$h0; fg_is_member=true; theme-fg_seasonal-banner__feb2026=OFF; theme-fg=dark; theme-fg_header-color=black; wp_automatewoo_session_started=1; wp_woocommerce_session_0cae6f5cb929d209043cb97f8c2eee44=279914%7C1780330966%7C1779812566%7C%24generic%24yT5TBnMt-05tsPeD0wWbHCrSC7qWlr3NlsfB2v8t"
 
def strip_html(x):
    if isinstance(x, str):
        return re.sub(r"<.*?>", "", x)
    return x

def _browser_headers(cookie_string: str) -> dict:
    return {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/137.0.0.0 Safari/537.36"
        ),
        "Accept": (
            "text/html,application/xhtml+xml,application/xml;"
            "q=0.9,image/avif,image/webp,*/*;q=0.8"
        ),
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate, br",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1",
        "Sec-Fetch-Dest": "document",
        "Sec-Fetch-Mode": "navigate",
        "Sec-Fetch-Site": "none",
        "Sec-Fetch-User": "?1",
        "Referer": "https://www.baseball-reference.com/",
        "cookie": cookie_string,
    }

s3 = boto3.client(
    "s3",
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    )
bucket = "sports-analytics-files"


(function() {
  var fn = function() {
    
    (function(root) {
      function now() {
        return new Date();
      }
    
      var force = false;
    
      if (typeof root._bokeh_onload_callbacks === "undefined" || force === true) {
        root._bokeh_onload_callbacks = [];
        root._bokeh_is_loading = undefined;
      }
    
      
      
    
      var element = document.getElementById("6ff378eb-45c3-48bd-a017-14557974b7a9");
        if (element == null) {
          console.warn("Bokeh: autoload.js configured with elementid '6ff378eb-45c3-48bd-a017-14557974b7a9' but no matching script tag was found.")
        }
      
    
      function run_callbacks() {
        try {
          root._bokeh_onload_callbacks.forEach(function(callback) {
            if (callback != null)
              callback();
          });
        } finally {
          delete root._bokeh_onload_callbacks
        }
        console.debug("Bokeh: all callbacks have finished");
      }
    
      function load_libs(css_urls, js_urls, callback) {
        if (css_urls == null) css_urls = [];
        if (js_urls == null) js_urls = [];
    
        root._bokeh_onload_callbacks.push(callback);
        if (root._bokeh_is_loading > 0) {
          console.debug("Bokeh: BokehJS is being loaded, scheduling callback at", now());
          return null;
        }
        if (js_urls == null || js_urls.length === 0) {
          run_callbacks();
          return null;
        }
        console.debug("Bokeh: BokehJS not loaded, scheduling load and callback at", now());
        root._bokeh_is_loading = css_urls.length + js_urls.length;
    
        function on_load() {
          root._bokeh_is_loading--;
          if (root._bokeh_is_loading === 0) {
            console.debug("Bokeh: all BokehJS libraries/stylesheets loaded");
            run_callbacks()
          }
        }
    
        function on_error() {
          console.error("failed to load " + url);
        }
    
        for (var i = 0; i < css_urls.length; i++) {
          var url = css_urls[i];
          const element = document.createElement("link");
          element.onload = on_load;
          element.onerror = on_error;
          element.rel = "stylesheet";
          element.type = "text/css";
          element.href = url;
          console.debug("Bokeh: injecting link tag for BokehJS stylesheet: ", url);
          document.body.appendChild(element);
        }
    
        const hashes = {"https://cdn.bokeh.org/bokeh/release/bokeh-2.2.3.min.js": "T2yuo9Oe71Cz/I4X9Ac5+gpEa5a8PpJCDlqKYO0CfAuEszu1JrXLl8YugMqYe3sM", "https://cdn.bokeh.org/bokeh/release/bokeh-widgets-2.2.3.min.js": "98GDGJ0kOMCUMUePhksaQ/GYgB3+NH9h996V88sh3aOiUNX3N+fLXAtry6xctSZ6", "https://cdn.bokeh.org/bokeh/release/bokeh-tables-2.2.3.min.js": "89bArO+nlbP3sgakeHjCo1JYxYR5wufVgA3IbUvDY+K7w4zyxJqssu7wVnfeKCq8"};
    
        for (var i = 0; i < js_urls.length; i++) {
          var url = js_urls[i];
          var element = document.createElement('script');
          element.onload = on_load;
          element.onerror = on_error;
          element.async = false;
          element.src = url;
          if (url in hashes) {
            element.crossOrigin = "anonymous";
            element.integrity = "sha384-" + hashes[url];
          }
          console.debug("Bokeh: injecting script tag for BokehJS library: ", url);
          document.head.appendChild(element);
        }
      };
    
      function inject_raw_css(css) {
        const element = document.createElement("style");
        element.appendChild(document.createTextNode(css));
        document.body.appendChild(element);
      }
    
      
      var js_urls = ["https://cdn.bokeh.org/bokeh/release/bokeh-2.2.3.min.js", "https://cdn.bokeh.org/bokeh/release/bokeh-widgets-2.2.3.min.js", "https://cdn.bokeh.org/bokeh/release/bokeh-tables-2.2.3.min.js"];
      var css_urls = [];
      
    
      var inline_js = [
        function(Bokeh) {
          Bokeh.set_log_level("info");
        },
        
        function(Bokeh) {
          (function() {
            var fn = function() {
              Bokeh.safely(function() {
                (function(root) {
                  function embed_document(root) {
                    
                  var docs_json = '{"93e0de49-9697-45b4-a8d2-3a37bbed25d9":{"roots":{"references":[{"attributes":{},"id":"17889","type":"UnionRenderers"},{"attributes":{"line_color":"#fa7c17","line_width":1.5,"x":{"field":"x"},"y":{"field":"y"}},"id":"17781","type":"Line"},{"attributes":{"fill_color":{"value":"#2a2eec"},"line_color":{"value":"#1f77b4"},"size":{"units":"screen","value":4.5},"x":{"value":3.4162688471038556},"y":{"value":2.55}},"id":"17851","type":"Circle"},{"attributes":{},"id":"17891","type":"UnionRenderers"},{"attributes":{"source":{"id":"17790"}},"id":"17794","type":"CDSView"},{"attributes":{},"id":"17867","type":"UnionRenderers"},{"attributes":{"fill_alpha":{"value":0.1},"fill_color":{"value":"#2a2eec"},"line_alpha":{"value":0.1},"line_color":{"value":"#1f77b4"},"size":{"units":"screen","value":4.5},"x":{"value":3.4162688471038556},"y":{"value":2.55}},"id":"17852","type":"Circle"},{"attributes":{},"id":"17879","type":"UnionRenderers"},{"attributes":{"data_source":{"id":"17750"},"glyph":{"id":"17751"},"hover_glyph":null,"muted_glyph":null,"nonselection_glyph":{"id":"17752"},"selection_glyph":null,"view":{"id":"17754"}},"id":"17753","type":"GlyphRenderer"},{"attributes":{"line_alpha":0.1,"line_color":"#2a2eec","line_width":1.5,"x":{"field":"x"},"y":{"field":"y"}},"id":"17842","type":"Line"},{"attributes":{"data_source":{"id":"17830"},"glyph":{"id":"17831"},"hover_glyph":null,"muted_glyph":null,"nonselection_glyph":{"id":"17832"},"selection_glyph":null,"view":{"id":"17834"}},"id":"17833","type":"GlyphRenderer"},{"attributes":{"data":{},"selected":{"id":"17906"},"selection_policy":{"id":"17905"}},"id":"17835","type":"ColumnDataSource"},{"attributes":{},"id":"17881","type":"UnionRenderers"},{"attributes":{"data_source":{"id":"17735"},"glyph":{"id":"17736"},"hover_glyph":null,"muted_glyph":null,"nonselection_glyph":{"id":"17737"},"selection_glyph":null,"view":{"id":"17739"}},"id":"17738","type":"GlyphRenderer"},{"attributes":{"data":{},"selected":{"id":"17900"},"selection_policy":{"id":"17899"}},"id":"17820","type":"ColumnDataSource"},{"attributes":{},"id":"17875","type":"UnionRenderers"},{"attributes":{"bounds":"auto","min_interval":1},"id":"17857","type":"DataRange1d"},{"attributes":{"source":{"id":"17775"}},"id":"17779","type":"CDSView"},{"attributes":{"fill_color":{"value":"#2a2eec"},"line_color":{"value":"#1f77b4"},"size":{"units":"screen","value":4.5},"x":{"value":3.374380781729923},"y":{"value":1.65}},"id":"17806","type":"Circle"},{"attributes":{},"id":"17704","type":"LinearScale"},{"attributes":{"data":{"x":[-1.9958943490877263,9.311550558002505],"y":[1.65,1.65]},"selected":{"id":"17890"},"selection_policy":{"id":"17889"}},"id":"17795","type":"ColumnDataSource"},{"attributes":{},"id":"17874","type":"Selection"},{"attributes":{},"id":"17899","type":"UnionRenderers"},{"attributes":{"data_source":{"id":"17785"},"glyph":{"id":"17786"},"hover_glyph":null,"muted_glyph":null,"nonselection_glyph":{"id":"17787"},"selection_glyph":null,"view":{"id":"17789"}},"id":"17788","type":"GlyphRenderer"},{"attributes":{},"id":"17907","type":"UnionRenderers"},{"attributes":{"line_alpha":0.1,"line_color":"#2a2eec","line_width":3.0,"x":{"field":"x"},"y":{"field":"y"}},"id":"17817","type":"Line"},{"attributes":{},"id":"17895","type":"UnionRenderers"},{"attributes":{},"id":"17901","type":"UnionRenderers"},{"attributes":{"line_alpha":0.1,"line_color":"#fa7c17","line_width":1.5,"x":{"field":"x"},"y":{"field":"y"}},"id":"17782","type":"Line"},{"attributes":{"data_source":{"id":"17815"},"glyph":{"id":"17816"},"hover_glyph":null,"muted_glyph":null,"nonselection_glyph":{"id":"17817"},"selection_glyph":null,"view":{"id":"17819"}},"id":"17818","type":"GlyphRenderer"},{"attributes":{"data":{"x":[1.9127415870254314,7.0411499662839026],"y":[2.25,2.25]},"selected":{"id":"17904"},"selection_policy":{"id":"17903"}},"id":"17830","type":"ColumnDataSource"},{"attributes":{},"id":"17886","type":"Selection"},{"attributes":{"data_source":{"id":"17850"},"glyph":{"id":"17851"},"hover_glyph":null,"muted_glyph":null,"nonselection_glyph":{"id":"17852"},"selection_glyph":null,"view":{"id":"17854"}},"id":"17853","type":"GlyphRenderer"},{"attributes":{"data":{"x":[0.9649316231388204,5.949680273009057],"y":[1.65,1.65]},"selected":{"id":"17892"},"selection_policy":{"id":"17891"}},"id":"17800","type":"ColumnDataSource"},{"attributes":{},"id":"17864","type":"BasicTickFormatter"},{"attributes":{"data_source":{"id":"17790"},"glyph":{"id":"17791"},"hover_glyph":null,"muted_glyph":null,"nonselection_glyph":{"id":"17792"},"selection_glyph":null,"view":{"id":"17794"}},"id":"17793","type":"GlyphRenderer"},{"attributes":{},"id":"17882","type":"Selection"},{"attributes":{},"id":"17910","type":"Selection"},{"attributes":{"line_color":"#2a2eec","line_width":1.5,"x":{"field":"x"},"y":{"field":"y"}},"id":"17841","type":"Line"},{"attributes":{"toolbars":[{"id":"17726"}],"tools":[{"id":"17716"},{"id":"17717"},{"id":"17718"},{"id":"17719"},{"id":"17720"},{"id":"17721"},{"id":"17722"},{"id":"17723"}]},"id":"17917","type":"ProxyToolbar"},{"attributes":{"data_source":{"id":"17825"},"glyph":{"id":"17826"},"hover_glyph":null,"muted_glyph":null,"nonselection_glyph":{"id":"17827"},"selection_glyph":null,"view":{"id":"17829"}},"id":"17828","type":"GlyphRenderer"},{"attributes":{"line_color":"#2a2eec","line_width":1.5,"x":{"field":"x"},"y":{"field":"y"}},"id":"17811","type":"Line"},{"attributes":{"data":{"x":[2.2921199846409115,6.479187946875487],"y":[0.8999999999999999,0.8999999999999999]},"selected":{"id":"17886"},"selection_policy":{"id":"17885"}},"id":"17785","type":"ColumnDataSource"},{"attributes":{},"id":"17865","type":"UnionRenderers"},{"attributes":{},"id":"17876","type":"Selection"},{"attributes":{"source":{"id":"17780"}},"id":"17784","type":"CDSView"},{"attributes":{},"id":"17878","type":"Selection"},{"attributes":{},"id":"17888","type":"Selection"},{"attributes":{"data":{"x":[-0.7842313478998125,9.985046696913068],"y":[2.55,2.55]},"selected":{"id":"17908"},"selection_policy":{"id":"17907"}},"id":"17840","type":"ColumnDataSource"},{"attributes":{"data_source":{"id":"17805"},"glyph":{"id":"17806"},"hover_glyph":null,"muted_glyph":null,"nonselection_glyph":{"id":"17807"},"selection_glyph":null,"view":{"id":"17809"}},"id":"17808","type":"GlyphRenderer"},{"attributes":{"line_alpha":0.1,"line_color":"#2a2eec","line_width":3.0,"x":{"field":"x"},"y":{"field":"y"}},"id":"17847","type":"Line"},{"attributes":{},"id":"17894","type":"Selection"},{"attributes":{"data_source":{"id":"17845"},"glyph":{"id":"17846"},"hover_glyph":null,"muted_glyph":null,"nonselection_glyph":{"id":"17847"},"selection_glyph":null,"view":{"id":"17849"}},"id":"17848","type":"GlyphRenderer"},{"attributes":{"text":"94.0% HDI"},"id":"17855","type":"Title"},{"attributes":{"line_alpha":0.1,"line_color":"#2a2eec","line_width":1.5,"x":{"field":"x"},"y":{"field":"y"}},"id":"17812","type":"Line"},{"attributes":{"data_source":{"id":"17835"},"glyph":{"id":"17836"},"hover_glyph":null,"muted_glyph":null,"nonselection_glyph":{"id":"17837"},"selection_glyph":null,"view":{"id":"17839"}},"id":"17838","type":"GlyphRenderer"},{"attributes":{},"id":"17883","type":"UnionRenderers"},{"attributes":{},"id":"17909","type":"UnionRenderers"},{"attributes":{"data_source":{"id":"17795"},"glyph":{"id":"17796"},"hover_glyph":null,"muted_glyph":null,"nonselection_glyph":{"id":"17797"},"selection_glyph":null,"view":{"id":"17799"}},"id":"17798","type":"GlyphRenderer"},{"attributes":{"line_color":"#2a2eec","line_width":3.0,"x":{"field":"x"},"y":{"field":"y"}},"id":"17801","type":"Line"},{"attributes":{"source":{"id":"17850"}},"id":"17854","type":"CDSView"},{"attributes":{},"id":"17885","type":"UnionRenderers"},{"attributes":{"below":[{"id":"17708"}],"center":[{"id":"17711"},{"id":"17715"}],"left":[{"id":"17712"}],"outline_line_color":null,"output_backend":"webgl","plot_height":435,"plot_width":360,"renderers":[{"id":"17738"},{"id":"17743"},{"id":"17748"},{"id":"17753"},{"id":"17758"},{"id":"17763"},{"id":"17768"},{"id":"17773"},{"id":"17778"},{"id":"17783"},{"id":"17788"},{"id":"17793"},{"id":"17798"},{"id":"17803"},{"id":"17808"},{"id":"17813"},{"id":"17818"},{"id":"17823"},{"id":"17828"},{"id":"17833"},{"id":"17838"},{"id":"17843"},{"id":"17848"},{"id":"17853"}],"title":{"id":"17855"},"toolbar":{"id":"17726"},"toolbar_location":null,"x_range":{"id":"17857"},"x_scale":{"id":"17704"},"y_range":{"id":"17858"},"y_scale":{"id":"17706"}},"id":"17699","subtype":"Figure","type":"Plot"},{"attributes":{"ticks":[0.44999999999999996,2.0999999999999996]},"id":"17859","type":"FixedTicker"},{"attributes":{},"id":"17902","type":"Selection"},{"attributes":{"fill_alpha":{"value":0.1},"fill_color":{"value":"#2a2eec"},"line_alpha":{"value":0.1},"line_color":{"value":"#1f77b4"},"size":{"units":"screen","value":4.5},"x":{"value":3.374380781729923},"y":{"value":1.65}},"id":"17807","type":"Circle"},{"attributes":{"line_color":"#2a2eec","line_width":1.5,"x":{"field":"x"},"y":{"field":"y"}},"id":"17796","type":"Line"},{"attributes":{},"id":"17893","type":"UnionRenderers"},{"attributes":{"data":{},"selected":{"id":"17894"},"selection_policy":{"id":"17893"}},"id":"17805","type":"ColumnDataSource"},{"attributes":{},"id":"17870","type":"Selection"},{"attributes":{"data_source":{"id":"17820"},"glyph":{"id":"17821"},"hover_glyph":null,"muted_glyph":null,"nonselection_glyph":{"id":"17822"},"selection_glyph":null,"view":{"id":"17824"}},"id":"17823","type":"GlyphRenderer"},{"attributes":{},"id":"17871","type":"UnionRenderers"},{"attributes":{},"id":"17884","type":"Selection"},{"attributes":{"children":[[{"id":"17699"},0,0]]},"id":"17916","type":"GridBox"},{"attributes":{"fill_alpha":{"value":0.1},"fill_color":{"value":"#fa7c17"},"line_alpha":{"value":0.1},"line_color":{"value":"#1f77b4"},"size":{"units":"screen","value":4.5},"x":{"value":4.59253463805363},"y":{"value":0.6}},"id":"17777","type":"Circle"},{"attributes":{},"id":"17904","type":"Selection"},{"attributes":{"source":{"id":"17800"}},"id":"17804","type":"CDSView"},{"attributes":{"data":{"x":[2.393576367885318,6.880909766712211],"y":[1.95,1.95]},"selected":{"id":"17898"},"selection_policy":{"id":"17897"}},"id":"17815","type":"ColumnDataSource"},{"attributes":{},"id":"17880","type":"Selection"},{"attributes":{"data_source":{"id":"17810"},"glyph":{"id":"17811"},"hover_glyph":null,"muted_glyph":null,"nonselection_glyph":{"id":"17812"},"selection_glyph":null,"view":{"id":"17814"}},"id":"17813","type":"GlyphRenderer"},{"attributes":{"fill_alpha":{"value":0.1},"fill_color":{"value":"#fa7c17"},"line_alpha":{"value":0.1},"line_color":{"value":"#1f77b4"},"size":{"units":"screen","value":4.5},"x":{"value":4.3706630373472235},"y":{"value":0.8999999999999999}},"id":"17792","type":"Circle"},{"attributes":{},"id":"17862","type":"BasicTickFormatter"},{"attributes":{},"id":"17887","type":"UnionRenderers"},{"attributes":{"fill_alpha":{"value":0.1},"fill_color":{"value":"#2a2eec"},"line_alpha":{"value":0.1},"line_color":{"value":"#1f77b4"},"size":{"units":"screen","value":4.5},"x":{"value":4.181352887007439},"y":{"value":2.25}},"id":"17837","type":"Circle"},{"attributes":{},"id":"17896","type":"Selection"},{"attributes":{"source":{"id":"17810"}},"id":"17814","type":"CDSView"},{"attributes":{},"id":"17890","type":"Selection"},{"attributes":{},"id":"17906","type":"Selection"},{"attributes":{"data_source":{"id":"17840"},"glyph":{"id":"17841"},"hover_glyph":null,"muted_glyph":null,"nonselection_glyph":{"id":"17842"},"selection_glyph":null,"view":{"id":"17844"}},"id":"17843","type":"GlyphRenderer"},{"attributes":{"line_color":"#2a2eec","line_width":3.0,"x":{"field":"x"},"y":{"field":"y"}},"id":"17831","type":"Line"},{"attributes":{"source":{"id":"17840"}},"id":"17844","type":"CDSView"},{"attributes":{"bounds":"auto","end":3.4499999999999997,"min_interval":2,"start":-0.8999999999999999},"id":"17858","type":"DataRange1d"},{"attributes":{"line_color":"#2a2eec","line_width":3.0,"x":{"field":"x"},"y":{"field":"y"}},"id":"17816","type":"Line"},{"attributes":{"source":{"id":"17825"}},"id":"17829","type":"CDSView"},{"attributes":{"data":{},"selected":{"id":"17912"},"selection_policy":{"id":"17911"}},"id":"17850","type":"ColumnDataSource"},{"attributes":{"fill_color":{"value":"#2a2eec"},"line_color":{"value":"#1f77b4"},"size":{"units":"screen","value":4.5},"x":{"value":4.181352887007439},"y":{"value":2.25}},"id":"17836","type":"Circle"},{"attributes":{},"id":"17892","type":"Selection"},{"attributes":{"data_source":{"id":"17775"},"glyph":{"id":"17776"},"hover_glyph":null,"muted_glyph":null,"nonselection_glyph":{"id":"17777"},"selection_glyph":null,"view":{"id":"17779"}},"id":"17778","type":"GlyphRenderer"},{"attributes":{},"id":"17872","type":"Selection"},{"attributes":{"line_alpha":0.1,"line_color":"#2a2eec","line_width":3.0,"x":{"field":"x"},"y":{"field":"y"}},"id":"17802","type":"Line"},{"attributes":{"data":{},"selected":{"id":"17888"},"selection_policy":{"id":"17887"}},"id":"17790","type":"ColumnDataSource"},{"attributes":{"source":{"id":"17845"}},"id":"17849","type":"CDSView"},{"attributes":{},"id":"17873","type":"UnionRenderers"},{"attributes":{"source":{"id":"17805"}},"id":"17809","type":"CDSView"},{"attributes":{},"id":"17866","type":"Selection"},{"attributes":{},"id":"17903","type":"UnionRenderers"},{"attributes":{"fill_color":{"value":"#fa7c17"},"line_color":{"value":"#1f77b4"},"size":{"units":"screen","value":4.5},"x":{"value":4.3706630373472235},"y":{"value":0.8999999999999999}},"id":"17791","type":"Circle"},{"attributes":{},"id":"17908","type":"Selection"},{"attributes":{"line_color":"#2a2eec","line_width":1.5,"x":{"field":"x"},"y":{"field":"y"}},"id":"17826","type":"Line"},{"attributes":{"source":{"id":"17830"}},"id":"17834","type":"CDSView"},{"attributes":{"data_source":{"id":"17800"},"glyph":{"id":"17801"},"hover_glyph":null,"muted_glyph":null,"nonselection_glyph":{"id":"17802"},"selection_glyph":null,"view":{"id":"17804"}},"id":"17803","type":"GlyphRenderer"},{"attributes":{"source":{"id":"17785"}},"id":"17789","type":"CDSView"},{"attributes":{"line_alpha":0.1,"line_color":"#fa7c17","line_width":3.0,"x":{"field":"x"},"y":{"field":"y"}},"id":"17787","type":"Line"},{"attributes":{"line_color":"#fa7c17","line_width":3.0,"x":{"field":"x"},"y":{"field":"y"}},"id":"17786","type":"Line"},{"attributes":{},"id":"17869","type":"UnionRenderers"},{"attributes":{"data":{"x":[-2.061896746944455,11.515205993744095],"y":[2.25,2.25]},"selected":{"id":"17902"},"selection_policy":{"id":"17901"}},"id":"17825","type":"ColumnDataSource"},{"attributes":{},"id":"17912","type":"Selection"},{"attributes":{"source":{"id":"17820"}},"id":"17824","type":"CDSView"},{"attributes":{"source":{"id":"17815"}},"id":"17819","type":"CDSView"},{"attributes":{"fill_color":{"value":"#2a2eec"},"line_color":{"value":"#1f77b4"},"size":{"units":"screen","value":4.5},"x":{"value":4.74129666559688},"y":{"value":1.95}},"id":"17821","type":"Circle"},{"attributes":{"line_alpha":0.1,"line_color":"#2a2eec","line_width":3.0,"x":{"field":"x"},"y":{"field":"y"}},"id":"17832","type":"Line"},{"attributes":{"data":{"x":[-2.358040118461496,10.910290467635015],"y":[1.95,1.95]},"selected":{"id":"17896"},"selection_policy":{"id":"17895"}},"id":"17810","type":"ColumnDataSource"},{"attributes":{},"id":"17868","type":"Selection"},{"attributes":{"line_alpha":0.1,"line_color":"#2a2eec","line_width":1.5,"x":{"field":"x"},"y":{"field":"y"}},"id":"17827","type":"Line"},{"attributes":{},"id":"17898","type":"Selection"},{"attributes":{},"id":"17900","type":"Selection"},{"attributes":{"data":{"x":[-2.2641273152868076,10.475301593619335],"y":[0.8999999999999999,0.8999999999999999]},"selected":{"id":"17884"},"selection_policy":{"id":"17883"}},"id":"17780","type":"ColumnDataSource"},{"attributes":{"line_color":"#2a2eec","line_width":3.0,"x":{"field":"x"},"y":{"field":"y"}},"id":"17846","type":"Line"},{"attributes":{},"id":"17877","type":"UnionRenderers"},{"attributes":{"fill_alpha":{"value":0.1},"fill_color":{"value":"#2a2eec"},"line_alpha":{"value":0.1},"line_color":{"value":"#1f77b4"},"size":{"units":"screen","value":4.5},"x":{"value":4.74129666559688},"y":{"value":1.95}},"id":"17822","type":"Circle"},{"attributes":{},"id":"17905","type":"UnionRenderers"},{"attributes":{"source":{"id":"17835"}},"id":"17839","type":"CDSView"},{"attributes":{"data":{"x":[1.9790083397469873,5.455951625836456],"y":[2.55,2.55]},"selected":{"id":"17910"},"selection_policy":{"id":"17909"}},"id":"17845","type":"ColumnDataSource"},{"attributes":{"line_alpha":0.1,"line_color":"#2a2eec","line_width":1.5,"x":{"field":"x"},"y":{"field":"y"}},"id":"17797","type":"Line"},{"attributes":{"source":{"id":"17795"}},"id":"17799","type":"CDSView"},{"attributes":{"data_source":{"id":"17740"},"glyph":{"id":"17741"},"hover_glyph":null,"muted_glyph":null,"nonselection_glyph":{"id":"17742"},"selection_glyph":null,"view":{"id":"17744"}},"id":"17743","type":"GlyphRenderer"},{"attributes":{"source":{"id":"17750"}},"id":"17754","type":"CDSView"},{"attributes":{},"id":"17716","type":"ResetTool"},{"attributes":{"fill_color":{"value":"#fa7c17"},"line_color":{"value":"#1f77b4"},"size":{"units":"screen","value":4.5},"x":{"value":4.409527442378456},"y":{"value":0.3}},"id":"17761","type":"Circle"},{"attributes":{"data_source":{"id":"17765"},"glyph":{"id":"17766"},"hover_glyph":null,"muted_glyph":null,"nonselection_glyph":{"id":"17767"},"selection_glyph":null,"view":{"id":"17769"}},"id":"17768","type":"GlyphRenderer"},{"attributes":{"line_alpha":0.1,"line_color":"#fa7c17","line_width":1.5,"x":{"field":"x"},"y":{"field":"y"}},"id":"17737","type":"Line"},{"attributes":{"data_source":{"id":"17760"},"glyph":{"id":"17761"},"hover_glyph":null,"muted_glyph":null,"nonselection_glyph":{"id":"17762"},"selection_glyph":null,"view":{"id":"17764"}},"id":"17763","type":"GlyphRenderer"},{"attributes":{"axis":{"id":"17712"},"dimension":1,"grid_line_color":null,"ticker":null},"id":"17715","type":"Grid"},{"attributes":{"active_drag":"auto","active_inspect":"auto","active_multi":null,"active_scroll":"auto","active_tap":"auto","tools":[{"id":"17716"},{"id":"17717"},{"id":"17718"},{"id":"17719"},{"id":"17720"},{"id":"17721"},{"id":"17722"},{"id":"17723"}]},"id":"17726","type":"Toolbar"},{"attributes":{"data":{},"selected":{"id":"17876"},"selection_policy":{"id":"17875"}},"id":"17760","type":"ColumnDataSource"},{"attributes":{"formatter":{"id":"17862"},"ticker":{"id":"17709"}},"id":"17708","type":"LinearAxis"},{"attributes":{},"id":"17722","type":"SaveTool"},{"attributes":{"data":{"x":[2.2808674873161356,6.6262125124574265],"y":[0.3,0.3]},"selected":{"id":"17874"},"selection_policy":{"id":"17873"}},"id":"17755","type":"ColumnDataSource"},{"attributes":{"fill_color":{"value":"#fa7c17"},"line_color":{"value":"#1f77b4"},"size":{"units":"screen","value":4.5},"x":{"value":4.663971371122512},"y":{"value":0}},"id":"17746","type":"Circle"},{"attributes":{},"id":"17897","type":"UnionRenderers"},{"attributes":{"source":{"id":"17745"}},"id":"17749","type":"CDSView"},{"attributes":{},"id":"17717","type":"PanTool"},{"attributes":{"line_color":"#fa7c17","line_width":1.5,"x":{"field":"x"},"y":{"field":"y"}},"id":"17736","type":"Line"},{"attributes":{"children":[{"id":"17918"},{"id":"17916"}]},"id":"17919","type":"Column"},{"attributes":{"overlay":{"id":"17725"}},"id":"17720","type":"LassoSelectTool"},{"attributes":{"overlay":{"id":"17724"}},"id":"17718","type":"BoxZoomTool"},{"attributes":{"data":{"x":[-1.4753067334040253,11.208804111875878],"y":[0.6,0.6]},"selected":{"id":"17878"},"selection_policy":{"id":"17877"}},"id":"17765","type":"ColumnDataSource"},{"attributes":{"data":{"x":[-2.13476994033137,10.008689721107494],"y":[0.3,0.3]},"selected":{"id":"17872"},"selection_policy":{"id":"17871"}},"id":"17750","type":"ColumnDataSource"},{"attributes":{"line_alpha":0.1,"line_color":"#fa7c17","line_width":1.5,"x":{"field":"x"},"y":{"field":"y"}},"id":"17767","type":"Line"},{"attributes":{"data":{},"selected":{"id":"17870"},"selection_policy":{"id":"17869"}},"id":"17745","type":"ColumnDataSource"},{"attributes":{"data_source":{"id":"17745"},"glyph":{"id":"17746"},"hover_glyph":null,"muted_glyph":null,"nonselection_glyph":{"id":"17747"},"selection_glyph":null,"view":{"id":"17749"}},"id":"17748","type":"GlyphRenderer"},{"attributes":{},"id":"17706","type":"LinearScale"},{"attributes":{"source":{"id":"17770"}},"id":"17774","type":"CDSView"},{"attributes":{"callback":null},"id":"17723","type":"HoverTool"},{"attributes":{},"id":"17719","type":"WheelZoomTool"},{"attributes":{"data":{},"selected":{"id":"17882"},"selection_policy":{"id":"17881"}},"id":"17775","type":"ColumnDataSource"},{"attributes":{"axis":{"id":"17708"},"ticker":null},"id":"17711","type":"Grid"},{"attributes":{"line_alpha":0.1,"line_color":"#fa7c17","line_width":3.0,"x":{"field":"x"},"y":{"field":"y"}},"id":"17742","type":"Line"},{"attributes":{"source":{"id":"17735"}},"id":"17739","type":"CDSView"},{"attributes":{"source":{"id":"17765"}},"id":"17769","type":"CDSView"},{"attributes":{"line_alpha":0.1,"line_color":"#fa7c17","line_width":3.0,"x":{"field":"x"},"y":{"field":"y"}},"id":"17757","type":"Line"},{"attributes":{"line_color":"#fa7c17","line_width":1.5,"x":{"field":"x"},"y":{"field":"y"}},"id":"17751","type":"Line"},{"attributes":{"fill_color":{"value":"#fa7c17"},"line_color":{"value":"#1f77b4"},"size":{"units":"screen","value":4.5},"x":{"value":4.59253463805363},"y":{"value":0.6}},"id":"17776","type":"Circle"},{"attributes":{"data_source":{"id":"17780"},"glyph":{"id":"17781"},"hover_glyph":null,"muted_glyph":null,"nonselection_glyph":{"id":"17782"},"selection_glyph":null,"view":{"id":"17784"}},"id":"17783","type":"GlyphRenderer"},{"attributes":{"source":{"id":"17740"}},"id":"17744","type":"CDSView"},{"attributes":{"formatter":{"id":"17864"},"major_label_overrides":{"0.44999999999999996":"Non Centered: mu","2.0999999999999996":"Centered: mu"},"ticker":{"id":"17859"}},"id":"17712","type":"LinearAxis"},{"attributes":{"fill_alpha":{"value":0.1},"fill_color":{"value":"#fa7c17"},"line_alpha":{"value":0.1},"line_color":{"value":"#1f77b4"},"size":{"units":"screen","value":4.5},"x":{"value":4.663971371122512},"y":{"value":0}},"id":"17747","type":"Circle"},{"attributes":{},"id":"17911","type":"UnionRenderers"},{"attributes":{"fill_alpha":0.5,"fill_color":"lightgrey","level":"overlay","line_alpha":1.0,"line_color":"black","line_dash":[4,4],"line_width":2,"xs_units":"screen","ys_units":"screen"},"id":"17725","type":"PolyAnnotation"},{"attributes":{},"id":"17721","type":"UndoTool"},{"attributes":{"line_alpha":0.1,"line_color":"#fa7c17","line_width":1.5,"x":{"field":"x"},"y":{"field":"y"}},"id":"17752","type":"Line"},{"attributes":{},"id":"17709","type":"BasicTicker"},{"attributes":{"data_source":{"id":"17755"},"glyph":{"id":"17756"},"hover_glyph":null,"muted_glyph":null,"nonselection_glyph":{"id":"17757"},"selection_glyph":null,"view":{"id":"17759"}},"id":"17758","type":"GlyphRenderer"},{"attributes":{"data":{"x":[-1.0798320889339172,10.200853218312133],"y":[0,0]},"selected":{"id":"17866"},"selection_policy":{"id":"17865"}},"id":"17735","type":"ColumnDataSource"},{"attributes":{"data_source":{"id":"17770"},"glyph":{"id":"17771"},"hover_glyph":null,"muted_glyph":null,"nonselection_glyph":{"id":"17772"},"selection_glyph":null,"view":{"id":"17774"}},"id":"17773","type":"GlyphRenderer"},{"attributes":{"toolbar":{"id":"17917"},"toolbar_location":"above"},"id":"17918","type":"ToolbarBox"},{"attributes":{"source":{"id":"17755"}},"id":"17759","type":"CDSView"},{"attributes":{"line_color":"#fa7c17","line_width":1.5,"x":{"field":"x"},"y":{"field":"y"}},"id":"17766","type":"Line"},{"attributes":{"data":{"x":[2.454317217804662,6.859709821773684],"y":[0,0]},"selected":{"id":"17868"},"selection_policy":{"id":"17867"}},"id":"17740","type":"ColumnDataSource"},{"attributes":{"fill_alpha":{"value":0.1},"fill_color":{"value":"#fa7c17"},"line_alpha":{"value":0.1},"line_color":{"value":"#1f77b4"},"size":{"units":"screen","value":4.5},"x":{"value":4.409527442378456},"y":{"value":0.3}},"id":"17762","type":"Circle"},{"attributes":{"bottom_units":"screen","fill_alpha":0.5,"fill_color":"lightgrey","left_units":"screen","level":"overlay","line_alpha":1.0,"line_color":"black","line_dash":[4,4],"line_width":2,"right_units":"screen","top_units":"screen"},"id":"17724","type":"BoxAnnotation"},{"attributes":{"line_color":"#fa7c17","line_width":3.0,"x":{"field":"x"},"y":{"field":"y"}},"id":"17756","type":"Line"},{"attributes":{"source":{"id":"17760"}},"id":"17764","type":"CDSView"},{"attributes":{"line_color":"#fa7c17","line_width":3.0,"x":{"field":"x"},"y":{"field":"y"}},"id":"17741","type":"Line"},{"attributes":{"data":{"x":[2.1521098502407368,6.9643589964055215],"y":[0.6,0.6]},"selected":{"id":"17880"},"selection_policy":{"id":"17879"}},"id":"17770","type":"ColumnDataSource"},{"attributes":{"line_color":"#fa7c17","line_width":3.0,"x":{"field":"x"},"y":{"field":"y"}},"id":"17771","type":"Line"},{"attributes":{"line_alpha":0.1,"line_color":"#fa7c17","line_width":3.0,"x":{"field":"x"},"y":{"field":"y"}},"id":"17772","type":"Line"}],"root_ids":["17919"]},"title":"Bokeh Application","version":"2.2.3"}}';
                  var render_items = [{"docid":"93e0de49-9697-45b4-a8d2-3a37bbed25d9","root_ids":["17919"],"roots":{"17919":"6ff378eb-45c3-48bd-a017-14557974b7a9"}}];
                  root.Bokeh.embed.embed_items(docs_json, render_items);
                
                  }
                  if (root.Bokeh !== undefined) {
                    embed_document(root);
                  } else {
                    var attempts = 0;
                    var timer = setInterval(function(root) {
                      if (root.Bokeh !== undefined) {
                        clearInterval(timer);
                        embed_document(root);
                      } else {
                        attempts++;
                        if (attempts > 100) {
                          clearInterval(timer);
                          console.log("Bokeh: ERROR: Unable to run BokehJS code because BokehJS library is missing");
                        }
                      }
                    }, 10, root)
                  }
                })(window);
              });
            };
            if (document.readyState != "loading") fn();
            else document.addEventListener("DOMContentLoaded", fn);
          })();
        },
        function(Bokeh) {
        
        
        }
      ];
    
      function run_inline_js() {
        
        for (var i = 0; i < inline_js.length; i++) {
          inline_js[i].call(root, root.Bokeh);
        }
        
      }
    
      if (root._bokeh_is_loading === 0) {
        console.debug("Bokeh: BokehJS loaded, going straight to plotting");
        run_inline_js();
      } else {
        load_libs(css_urls, js_urls, function() {
          console.debug("Bokeh: BokehJS plotting callback run at", now());
          run_inline_js();
        });
      }
    }(window));
  };
  if (document.readyState != "loading") fn();
  else document.addEventListener("DOMContentLoaded", fn);
})();